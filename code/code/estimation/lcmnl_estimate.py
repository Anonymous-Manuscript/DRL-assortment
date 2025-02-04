from numpy import array
from estimation.EMestimator import ExpectationMaximizationEstimator
from estimation.estimator import Estimator
from models.latent_class import LatentClassModel
from models.multinomial import MultinomiallogitModel
from estimation.optimization import NonLinearProblem,NonLinearSolver,Constraints,NonLinearProblem,Settings
from estimation.profiler import Profiler
import copy
from numpy import array, ones, zeros
import time
from collections import defaultdict
from utils import safe_log,ZERO_LOWER_BOUND,ONE_UPPER_BOUND,time_for_optimization


class LatentClassExpectationMaximizationEstimator(ExpectationMaximizationEstimator):
    """
        Taken from "Discrete Choice Methods with Simulation" by Kenneth E. Train (Second Edition Chapter 14).
    """
    def one_step(self, model, transactions):
        total_weights = 0.0
        weights = []

        lc_cache = {}
        for klass_share, klass_model in zip(model.gammas, model.mnl_models()):
            klass_transactions_weights = []
            mnl_cache = {}
            for transaction in transactions:
                memory = (transaction.product, tuple(transaction.offered_products))
                if memory in lc_cache:
                    lc_probability = lc_cache[memory]
                else:
                    lc_probability = model.probability_of(transaction)
                    lc_cache[memory] = lc_probability

                if memory in mnl_cache:
                    mnl_probability = mnl_cache[memory]
                else:
                    mnl_probability = klass_model.probability_of(transaction)
                    mnl_cache[memory] = mnl_probability

                numerator = (klass_share * mnl_probability)
                denominator = lc_probability
                probability = numerator / denominator
                total_weights += probability
                klass_transactions_weights.append(probability)
            weights.append(klass_transactions_weights)

        new_gammas = []
        for klass_transactions_weights in weights:
            new_gammas.append(sum(klass_transactions_weights) / total_weights)

        new_models = []
        for klass_transactions_weights, klass_model in zip(weights, model.mnl_models()):
            initial = copy.deepcopy(klass_model)
            problem = WeightedMultinomialLogitMaximumLikelihoodNonLinearProblem(initial, transactions,
                                                                                klass_transactions_weights)
            solution = NonLinearSolver.default().solve(problem, Profiler(verbose=False))
            initial.update_parameters_from_vector(solution)
            new_models.append(initial)

        return LatentClassModel(products=model.products, gammas=new_gammas, multi_logit_models=new_models)


class WeightedMultinomialLogitMaximumLikelihoodNonLinearProblem(NonLinearProblem):
    def __init__(self, model, transactions, transactions_weights):
        self.model = model
        self.transactions = transactions
        self.transaction_weights = transactions_weights

    def constraints(self):
        return self.model.constraints()

    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        result = 0.0
        cache = {}
        for weight, transaction in zip(self.transaction_weights, self.transactions):
            memory = (transaction.product, tuple(transaction.offered_products))
            if memory in cache:
                log_probability = cache[memory]
            else:
                log_probability = self.model.log_probability_of(transaction)
                cache[memory] = log_probability
            result += (weight * log_probability)
        return -result / len(self.transactions)

    def initial_solution(self):
        return array(self.model.parameters_vector())

    def amount_of_variables(self):
        return len(self.model.parameters_vector())


##################################################################################################################################
from func import transform_transaction_for_lcmnl
from scipy.optimize import minimize, fminbound
import time
import multiprocess as mp
from IPython import embed
from functools import partial
import numpy as np

# constants
NUM_BFGS_INIT = 20
MAX_CORRECTIVE_STEPS = 400
CORRECTIVE_GAP = 1e-8
NUM_SUBPROB_ITERS = 5
LOSS_DECREASE_TOL = 1e-4
class FrankWolfeMNLMixEst():
    """
    Runs the Frank-Wolfe algorithm to learn a latent class logit model
    from sales transaction data
    """
    def __init__(self, args, loss='likelihood', fw_variant='corrective', learning_rate=1.0, num_subprob_init=NUM_BFGS_INIT, regularization=False,
                 alpha=1.0):
        self.args = args
        
        # the loss function to optimize (negative log-likelihood or squared)
        self.loss = loss

        # determine which frank-wolfe variant to run
        # fixed-step-size: the classical Frank-Wolfe algorithm that uses a fixed step size in each iteration
        # linesearch: do a line search to choose the best step size in each iteration
        # corrective: fully corrective Frank-Wolfe variant that re-optimizes the objective function in each iteration
        self.fwVariant = fw_variant

        # the learning rate (used ONLY in the fixed-step-size variant)
        self.learning_rate = learning_rate

        # number of random starting points to run BFGS from in subproblems
        self.num_subprob_init = num_subprob_init

        # k x n numpy array where each row contains MNL coefficients for a class (k is # classes and n is # products)
        # first product is no-purchase so its coef is set to zero in all classes
        self.coefs_ = None
        # kx1 numpy array containing proportions of each class (k is # classes)
        self.mix_props = None

        # whether to have L2-regularization when solving the subproblems
        self.regularization = regularization
        # penalty term in L2-regularization
        self.alpha = alpha

    def _negative_log_likelihood(self, xk, n_counts):
        return -np.dot(n_counts, np.where(xk > 0, np.log(xk), 0))

    def _gradient_likelihood(self, xk, n_counts):
        return (-1.*n_counts)/xk

    def compute_optimization_objective(self, xk, n_counts, aux_info=None):
        if self.loss == 'squared':
            if aux_info is not None:
                return np.sum((xk*aux_info - n_counts)**2)/2
            else:
                return np.sum((xk - n_counts)**2)/2
        elif self.loss == 'likelihood':
            return self._negative_log_likelihood(xk, n_counts)

    def compute_objective_gradient(self, xk, n_counts, aux_info=None):
        if self.loss == 'squared':
            if aux_info is not None:
                return (aux_info*xk - n_counts)*aux_info
            else:
                return xk - n_counts
        elif self.loss == 'likelihood':
            return self._gradient_likelihood(xk, n_counts)

    def _predict_MNL_proba(self, membership, prod_coefs):
        prod_utils = prod_coefs - np.max(prod_coefs)
        prod_wt = np.exp(prod_utils)
        # compute unnormalized product weights
        probs = membership * prod_wt
        row_sums = np.sum(probs, 1)
        non_empty_assorts = (row_sums != 0)
        row_sums[~non_empty_assorts] = 1
        # compute choice probabilities
        probs = probs / row_sums[:, np.newaxis]
        return probs

    # predict choice probabilities of chosen products under current mixture parameters
    def _predict_choice_proba(self, mix_props, mix_coefs, membership, chosen_products):
        num_offer_sets = membership.shape[0]
        chosen_probs = np.zeros(num_offer_sets)
        for k in range(mix_props.shape[0]):
            chosen_probs += mix_props[k] * self._predict_MNL_proba(membership, mix_coefs[k])[range(num_offer_sets), chosen_products]
        return chosen_probs

    # predict choice probabilities for different offer-sets under estimated LC-MNL model
    # USE this function for evaluating out-of-sample performance
    def predict_choice_proba(self, membership):
        """
        membership: m X n binary matrix for m offersets and n products
        each row corresponds to offerset and each col corresponds to product
        membership[i, j] is 1 if product j is offered in offerset i
        output: choice_probs (m X n matrix) such that choice_probs[i, j] is probability of
        choosing product j from offerset i
        """
        choice_probs = np.zeros(membership.shape)
        for k in range(self.mix_props.shape[0]):
            choice_probs += self.mix_props[k] * self._predict_MNL_proba(membership, self.coefs_[k])

        assert np.all(np.around((np.sum(choice_probs, 1) - np.ones(choice_probs.shape[0])), 7) == 0), 'Choice probabilities not summing to 1'
        return choice_probs

    # Generic BFGS approach for solving the subproblem ###

    # compute subproblem objective and gradient with respect to params x0
    def FW_MNL_subprob_objective(self, x0, membership, num_sales, chosen_products):
        """
        x0 is only of length (n-1) where n is number of products
        coef corresponding to product 0 (no-purchase option) is always set to zero
        """
        num_offer_sets = membership.shape[0]
        probs = self._predict_MNL_proba(membership, np.insert(x0, 0, 0))
        chosen_probs = probs[range(num_offer_sets), chosen_products]
        if np.any(chosen_probs <= 0):
            return 1e10, -np.ones_like(x0)
        weighted_sales = num_sales * chosen_probs
        obj = np.sum(weighted_sales)
        chosen_prods_matrix = np.zeros_like(membership)
        chosen_prods_matrix[range(num_offer_sets), chosen_products] = 1
        grad_vec = weighted_sales[:, np.newaxis] * (chosen_prods_matrix - probs)
        grad = np.sum(grad_vec, axis=0)[1:]
        if self.regularization:
            grad += self.alpha * x0
            obj += .5 * self.alpha * (np.linalg.norm(x0) ** 2)
        if np.any(np.isnan(grad)):
            embed()

        grad[np.abs(grad) < 1e-15] = 0.
        return obj, grad

    def _base_subproblems(self, point_info, X_obs, gradk, C_obs):
        start_point = point_info[1]
        return minimize(self.FW_MNL_subprob_objective, start_point, args=(X_obs, gradk, C_obs), method='BFGS', jac=True, options={'disp': False}), point_info[0]

    # ==============================================================================

    # top-level function for solving the subproblem in each iteration of the Frank Wolfe algorithm
    def _FW_iteration(self, X_obs, xk, gradk, C_obs, curr_iter):
        num_params = X_obs.shape[1] - 1 # ignore no-purchase coef as it is set to zero
        for num_tries in range(NUM_SUBPROB_ITERS):
            # determine starting points for BFGS-based solution to each iteration
            cand_start_points = np.random.randn(self.num_subprob_init, num_params)
            # run BFGS from different starting points
            pool = mp.Pool(processes=5)
            results = pool.map(partial(self._base_subproblems, X_obs=X_obs, gradk=gradk, C_obs=C_obs), enumerate(cand_start_points))
            pool.close()
            pool.join()
            # compute the best solution
            best_result = min(results, key=lambda w: w[0].fun)
            param_vector = best_result[0].x
            param_vector = np.insert(param_vector, 0, 0)
            # compute choice probs under best param vector
            next_best_probs = self._predict_MNL_proba(X_obs, param_vector)[range(X_obs.shape[0]), C_obs]
            # compute next frank-wolfe direction
            next_dir = next_best_probs - xk
            # check if descent direction
            if np.dot(gradk, next_dir) < 0:
                return param_vector, next_best_probs
                # else try again

        # could not find a descent direction !
        print.warning('Could not find an improving solution at iteration %d for variant %s. Check if optimal solution has already been reached.', curr_iter, self.fwVariant, )
        return None, None
    # ==================================================================================================
    # FUNCTIONS FOR PERFORMING FULLY CORRECTIVE VARIANT of FRANK-WOLFE (FCFW)
    # ==================================================================================================

    # helper method to check if current solution lies within convex hull of vertices
    def _check_convex_combination(self, xk, alphas, vertices):
        result = np.abs(np.sum(alphas * vertices, 1) - xk).sum()
        assert result < 1e-8, 'Not a convex combination:' + str(result)
        assert 1 - np.sum(alphas) < 1e-10, 'Sum to 1 violated'

    # outer wrapper for performing FCFW
    def _perform_fully_corrective_step(self, X_obs, x_init, n_counts, alpha_init, max_iter, C_obs):
        num_support = self.coefs_.shape[0]
        num_data_points = X_obs.shape[0]
        soln = np.copy(x_init)
        alpha_coord = np.copy(alpha_init)

        # compute current objective value
        prev_obj = self.compute_optimization_objective(soln, n_counts)
        # compute the vertices in the polytope
        curr_prob_matrix = np.zeros((num_data_points, num_support))
        for k in range(num_support):
            curr_prob_matrix[:, k] = self._predict_MNL_proba(X_obs, self.coefs_[k])[range(num_data_points), C_obs]

        # perform the correction steps
        for iter in range(max_iter):
            # check if soln is in the feasible
            self._check_convex_combination(soln, alpha_coord, curr_prob_matrix)
            # compute current gradient
            curr_grad = self.compute_objective_gradient(soln, n_counts)
            # compute FW vertex and direction
            fw_weights = np.dot(curr_grad, curr_prob_matrix)
            fw_vertex = np.argmin(fw_weights)
            fw_direction = (curr_prob_matrix[:, fw_vertex] - soln)

            # compute away vertex among vertices with non-zero weight
            away_weights = np.where(alpha_coord > 0, fw_weights, -np.inf)
            away_vertex = np.argmax(away_weights)
            away_direction = (soln - curr_prob_matrix[:, away_vertex])

            # check duality gap
            gap = np.dot(-1 * curr_grad, fw_direction + away_direction)
            if gap < CORRECTIVE_GAP:
                break

            # do an away step
            if np.dot(-1 * curr_grad, fw_direction) < np.dot(-1 * curr_grad, away_direction) and (away_vertex != fw_vertex) and alpha_coord[away_vertex] < 1:
                away_step = True
                dirk = away_direction
                gamma_max = alpha_coord[away_vertex] / (1 - alpha_coord[away_vertex])
                # assert (gamma_max > 0), 'alpha of away vertex is %.8f for variant %s' %(alpha_coord[away_vertex], self.fwVariant)
            else:
                # do a frank-wolfe step
                away_step = False
                dirk = fw_direction
                gamma_max = 1.

            # do line search to compute step size
            opt_step_size = self._perform_line_search_step(soln, dirk, n_counts, gamma_max)

            # update barycentric coordinates alpha
            if not away_step:
                alpha_coord *= (1 - opt_step_size)
                alpha_coord[fw_vertex] += opt_step_size
            else:
                alpha_coord *= (1 + opt_step_size)
                alpha_coord[away_vertex] -= opt_step_size

            # update current solution
            soln += opt_step_size * dirk
            # clip alphas below precision
            alpha_coord[alpha_coord < 1e-15] = 0.
            # update objective value
            curr_obj = self.compute_optimization_objective(soln, n_counts)
            prev_obj = curr_obj

        print('Performed %d corrective steps', iter)
        # update weights of mixture components
        self.mix_props = alpha_coord
        # return current estimate of soln
        return soln

    # =================================================================
    # LINESEARCH ROUTINES
    # =================================================================

    def _brent_line_search(self, alpha, curr_probs, next_dir, n_counts, aux_info):
        return self.compute_optimization_objective(curr_probs + alpha * next_dir, n_counts, aux_info)

    def _perform_line_search_step(self, xk, dk, n_counts, upper_bound=1, aux_info=None):
        return fminbound(self._brent_line_search, 0, upper_bound, args=(xk, dk, n_counts, aux_info), xtol=1e-8)

    # ================================================
    # MAIN FRANK - WOLFE CODE
    # ================================================

    # outer wrapper for learning LC-MNL model from choice data
    def fit_to_choice_data(self, train_data, num_iters, init_coefs=None, init_mix_props=None):
        membership_train, sales_train = transform_transaction_for_lcmnl(self.args, train_data)
        num_os, num_prods = membership_train.shape
        prods_chosen = sales_train.nonzero()[1]#
        n_counts = sales_train[sales_train > 0]#
        n_obs_per_offerset = np.sum(sales_train > 0, 1)
        membership = np.repeat(membership_train, n_obs_per_offerset, axis=0)#
        """
        membership: m X n binary matrix for m offersets and n products
        each row corresponds to offerset and each col corresponds to product
        membership[i, j] is 1 if product j is offered in offerset i
        prods_chosen: array of length m
        prods_chosen[i] specifies the product chosen in offerset i
        n_counts: array of length m
        n_counts[i] specifies number of sales of product <prods_chosen[i]> when offered in offerset i
        IMPORTANT: the input should be aggregated to ensure that for a given offerset, each chosen product appears only once in the prods_chosen array
        num_iters: number of iterations of the Frank-Wolfe algorithm
        init_coefs: coefs to initialize with
        init_mix_props: if initializing with more than one class, the proportions of each class
        """
        num_os, num_prods = membership.shape
        start = time.time()
        if init_coefs is not None:
            self.coefs_ = np.atleast_2d(init_coefs.copy())
        else:
            self.coefs_ = np.zeros((1, num_prods))
            assert init_mix_props is None, 'Initial coefs not provided'
        init_num_classes = self.coefs_.shape[0]
        self.mix_props = np.ones(init_num_classes)/init_num_classes if init_mix_props is None else np.array(init_mix_props)
        curr_probs = self._predict_choice_proba(self.mix_props, self.coefs_, membership, prods_chosen)
        # compute the starting objective value
        prev_obj = self.compute_optimization_objective(curr_probs, n_counts)
        # relative change in the loss objective (can also be used as a stopping criterion)
        rel_change_in_loss = 1

        for iter in range(num_iters):
            print('Starting search for iteration:%d with rel_change_in_loss: %.4f and number of components: %d', iter + 1, rel_change_in_loss, np.count_nonzero(self.mix_props))
            (next_param_vector, next_probs) = self._FW_iteration(membership, curr_probs, self.compute_objective_gradient(curr_probs, n_counts), prods_chosen, iter+1)
            if next_param_vector is None:
                break
            # compute frank wolfe direction
            shiftedFWdir = next_probs - curr_probs

            # find the optimal step size
            if 'fixed-step-size' in self.fwVariant:
                step_size = 2 * self.learning_rate / (iter + 3)
            else:
                step_size = self._perform_line_search_step(curr_probs, shiftedFWdir, n_counts)
            # initialize correction to update using line search step size
            temp_probs = curr_probs + step_size * shiftedFWdir
            # initialize correction weights
            temp_weights = (1 - step_size) * self.mix_props
            # check if found component exists already
            param_indices = np.where((self.coefs_ == next_param_vector).all(axis=1))[0]
            if len(param_indices) > 0:
                temp_weights[param_indices[0]] += step_size
            else:
                # add the new component
                self.coefs_ = np.append(self.coefs_, next_param_vector[np.newaxis], 0)
                temp_weights = np.append(temp_weights, step_size)

            # Run FCFW on the components found so far
            if 'corrective' in self.fwVariant:
                curr_probs = self._perform_fully_corrective_step(membership, temp_probs, n_counts, temp_weights, MAX_CORRECTIVE_STEPS, prods_chosen)
            else:
                curr_probs = temp_probs
                self.mix_props = temp_weights

            curr_obj = self.compute_optimization_objective(curr_probs, n_counts)

            print('At iteration %d, current loss is %.4f for variant %s', iter + 1, curr_obj, self.fwVariant)
            if 'fixed-step-size' not in self.fwVariant:
                assert (curr_obj <= prev_obj or curr_obj - prev_obj < LOSS_DECREASE_TOL), 'Loss objective not decreasing at iteration %d for variant %s. Try increasing the value of the LOSS_DECREASE_TOL constant.' % (iter+1, self.fwVariant)
            rel_change_in_loss = np.abs(curr_obj - prev_obj) / np.abs(prev_obj)
            prev_obj = curr_obj

        print('Final loss is %.4f for variant %s after %.2f seconds', curr_obj, self.fwVariant, time.time() - start)











































##################################################################################################################################

class LatentClassFrankWolfeEstimator(Estimator):
    def likelihood_loss_function_coefficients(self, transactions):
        sales_per_transaction = defaultdict(lambda: 0.0)
        for transaction in transactions:
            sales_per_transaction[transaction] += 1.0
        return [(transaction, amount_of_sales) for transaction, amount_of_sales in list(sales_per_transaction.items())]

    def look_for_new_mnl_model(self, model, likelihood_loss_function_coefficients):
        possible_mnl_model = MultinomiallogitModel.simple_deterministic(model.products)
        problem = NewMNLSubProblem(model, possible_mnl_model, likelihood_loss_function_coefficients)
        solution = NonLinearSolver.default().solve(problem, self.profiler())
        possible_mnl_model.update_parameters_from_vector(solution)
        return possible_mnl_model

    def update_weights_for(self, model, likelihood_loss_function_coefficients):
        problem = NewWeightsSubProblem(model, likelihood_loss_function_coefficients)
        solution = NonLinearSolver.default().solve(problem, self.profiler())
        model.update_gammas_from(solution)

    def estimate(self, model, transactions):
        likelihood_loss_function_coefficients = self.likelihood_loss_function_coefficients(transactions)
        new_likelihood = model.log_likelihood_for(transactions)

        max_iterations = len(likelihood_loss_function_coefficients)
        cpu_time = time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                         total_time=Settings.instance().solver_total_time_limit(),
                                         profiler=self.profiler())
        start_time = time.time()

        for _ in range(max_iterations):
            old_likelihood = new_likelihood

            possible_mnl_model = self.look_for_new_mnl_model(model, likelihood_loss_function_coefficients)
            model.add_new_class_with(possible_mnl_model)
            self.update_weights_for(model, likelihood_loss_function_coefficients)

            new_likelihood = model.log_likelihood_for(transactions)

            likelihood_does_not_increase = new_likelihood < old_likelihood
            likelihood_does_not_increase_enough = abs(new_likelihood - old_likelihood) / len(transactions) < 1e-7
            time_limit = (time.time() - start_time) > cpu_time

            if likelihood_does_not_increase or likelihood_does_not_increase_enough or time_limit:
                break

        return model


class NewMNLSubProblem(NonLinearProblem):
    def __init__(self, latent_class_model, possible_mnl_model, likelihood_loss_function_coefficients):
        self.latent_class_model = latent_class_model
        self.likelihood_loss_function_coefficients = likelihood_loss_function_coefficients
        self.likelihood_loss_function_gradient = self.compute_likelihood_loss_function_gradient()
        self.possible_mnl_model = possible_mnl_model

    def compute_likelihood_loss_function_gradient(self):
        gradient = []
        for transaction, number_sales in self.likelihood_loss_function_coefficients:
            probability = self.latent_class_model.probability_of(transaction)
            gradient.append((transaction, - (number_sales / probability)))
        return gradient

    def constraints(self):
        return self.possible_mnl_model.constraints()

    def objective_function(self, parameters):
        self.possible_mnl_model.update_parameters_from_vector(parameters)
        result = 0
        for transaction, gradient_component in self.likelihood_loss_function_gradient:
            result += (gradient_component * self.possible_mnl_model.probability_of(transaction))
        return result / len(self.likelihood_loss_function_coefficients)

    def amount_of_variables(self):
        return len(self.possible_mnl_model.parameters_vector())

    def initial_solution(self):
        return array(self.possible_mnl_model.parameters_vector())


class NewWeightsSubProblem(NonLinearProblem):
    def __init__(self, model, likelihood_loss_function_coefficients):
        self.model = model
        self.likelihood_loss_function_coefficients = likelihood_loss_function_coefficients

    def constraints(self):
        return NewWeightsConstraints(self.model)

    def objective_function(self, vector):
        self.model.update_gammas_from(vector)
        result = 0.0
        for transaction, number_sales in self.likelihood_loss_function_coefficients:
            result -= (number_sales * safe_log(self.model.probability_of(transaction)))
        return result / len(self.likelihood_loss_function_coefficients)

    def amount_of_variables(self):
        return self.model.amount_of_classes()

    def initial_solution(self):
        return array(self.model.gammas)


class NewWeightsConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(self.model.amount_of_classes()) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(self.model.amount_of_classes()) * ONE_UPPER_BOUND

    def amount_of_constraints(self):
        return 1

    def lower_bounds_over_constraints_vector(self):
        return array([1.0])

    def upper_bounds_over_constraints_vector(self):
        return array([1.0])

    def non_zero_parameters_on_constraints_jacobian(self):
        return self.model.amount_of_classes()

    def constraints_evaluator(self):
        def evaluator(x):
            return array([sum(x)])
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                return (zeros(len(self.model.gammas)),
                        array(list(range(len(self.model.gammas)))))
            else:
                return ones(len(self.model.gammas))

        return jacobian_evaluator
