from estimation.optimization import Settings
from numpy.linalg import linalg
import time
from utils import time_for_optimization
NLP_LOWER_BOUND_INF = -1e19
NLP_UPPER_BOUND_INF = 1e19
ACCEPTABLE_ITERATIONS = 5
ACCEPTABLE_OBJ_DIFFERENCE = 1e-6
BUDGET_TIME_LIMIT = 60 * 30

class ConvergenceCriteria(object):
    def would_stop_this(self, profiler):
        raise NotImplementedError('Subclass responsibility')

    def reset_for(self, profiler):
        pass


class ObjectiveValueCriteria(ConvergenceCriteria):
    def __init__(self, acceptable_iterations, acceptable_objective_difference):
        self._acceptable_iterations = acceptable_iterations
        self._acceptable_objective_difference = acceptable_objective_difference
        self._last_considered_iteration = 0

    def acceptable_iterations(self):
        return self._acceptable_iterations

    def acceptable_objective_difference(self):
        return self._acceptable_objective_difference

    def reset_for(self, profiler):
        self._last_considered_iteration = len(profiler.iterations())

    def would_stop_this(self, profiler):
        last_iterations = profiler.iterations()[self._last_considered_iteration:][-self.acceptable_iterations():]
        if len(last_iterations) == self.acceptable_iterations():
            differences = [abs(last_iterations[i].value() - last_iterations[i - 1].value()) for i in range(1, len(last_iterations))]
            return all([difference < self.acceptable_objective_difference() for difference in differences])
        return False


class TimeBudgetCriteria(ConvergenceCriteria):
    def __init__(self, time_limit):
        """
        time_limit: Time limit in seconds
        """
        self._time_limit = time_limit

    def time_limit(self):
        return self._time_limit

    def would_stop_this(self, profiler):
        return profiler.duration() > self.time_limit()


class MixedConvergenceCriteria(ConvergenceCriteria):
    def __init__(self, criteria):
        self._criteria = criteria

    def reset(self):
        for criteria in self._criteria:
            criteria.reset()

    def would_stop_this(self, profiler):
        return any([criteria.would_stop_this(profiler) for criteria in self._criteria])


class Iteration(object):
    def __init__(self):
        self._start_time = time.time()
        self._stop_time = None
        self._value = None

    def is_finished(self):
        return self._value is not None

    def finish_with(self, value):
        if self.is_finished():
            raise Exception('Finishing already finished iteration.')
        self._value = value
        self._stop_time = time.time()

    def value(self):
        return self._value

    def start_time(self):
        return self._start_time

    def stop_time(self):
        return self._stop_time

    def duration(self):
        return self.stop_time() - self.start_time()

    def as_json(self):
        return {'start': self.start_time(),
                'stop': self.stop_time(),
                'value': self.value()}

    def __repr__(self):
        data = (self.start_time(), self.stop_time(), self.duration(), self.value())
        return '< Start: %s ; Stop: %s ; Duration %s ; Value: %s >' % data


class Profiler(object):
    def __init__(self, verbose=True):
        self._verbose = verbose
        self._iterations = []
        time_criteria = TimeBudgetCriteria(BUDGET_TIME_LIMIT)
        objective_value_criteria = ObjectiveValueCriteria(ACCEPTABLE_ITERATIONS, ACCEPTABLE_OBJ_DIFFERENCE)
        self._convergence_criteria = MixedConvergenceCriteria(criteria=[time_criteria, objective_value_criteria])

    def iterations(self):
        return self._iterations

    def convergence_criteria(self):
        return self._convergence_criteria

    def json_iterations(self):
        return [i.as_json() for i in self.iterations()]

    def last_iteration(self):
        return self._iterations[-1]

    def first_iteration(self):
        return self._iterations[0]

    def start_iteration(self):
        self._iterations.append(Iteration())

    def stop_iteration(self, value):
        self.last_iteration().finish_with(value)
        self.show_progress()

    def show_progress(self):
        if self._verbose:
            if len(self.iterations()) % 10 == 1: 
                a = 1
                #print('----------------------')
                #print('N#  \tTIME \tOBJ VALUE')
            #print(('%s\t%ss\t%.8f' % (len(self.iterations()), int(self.duration()), self.last_iteration().value())))

    def duration(self):
        if len(self.iterations()) > 0:
            return self.last_iteration().stop_time() - self.first_iteration().start_time()
        return 0

    def should_stop(self):
        return self.convergence_criteria().would_stop_this(self)

    def reset_convergence_criteria(self):
        self.convergence_criteria().reset_for(self)

    def update_time(self):
        if len(self._iterations) > 2:
            self.start_iteration()
            self.stop_iteration(self._iterations[-2].value())
class Estimator(object):
    """
        Estimates a model parameters based on historical transactions data.
    """
    def __init__(self):
        self._profiler = Profiler()

    def profiler(self):
        return self._profiler

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')



class ExpectationMaximizationEstimator(Estimator):
    def estimate(self, model, transactions):
        self.profiler().reset_convergence_criteria()
        self.profiler().update_time()
        model = self.custom_initial_solution(model, transactions)
        cpu_time = time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                         total_time=Settings.instance().solver_total_time_limit(),
                                         profiler=self.profiler())

        start_time = time.time()
        while True:
            self.profiler().start_iteration()
            model = self.one_step(model, transactions)
            likelihood = model.log_likelihood_for(transactions)
            self.profiler().stop_iteration(likelihood)

            if self.profiler().should_stop() or (time.time() - start_time) > cpu_time:
                break

        return model

    def one_step(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def custom_initial_solution(self, model, transactions):
        return model

class MarkovChainExpectationMaximizationEstimator(ExpectationMaximizationEstimator):
    def one_step(self, model, transactions):
        X, F = self.expectation_step(model, transactions)
        return self.maximization_step(model, X, F)

    def expectation_step(self, model, transactions):
        # Precalculate psis and thetas.
        psis = []
        thetas = []
        for transaction in transactions:
            psis.append(self.compute_psi(model, transaction))
            thetas.append(model.expected_number_of_visits_if(transaction.offered_products))

        F = self.estimate_F(model, transactions, psis, thetas)
        X = self.estimate_X(model, transactions, psis, thetas)
        return X, F

    def maximization_step(self, model, X, F):
        new_l = []
        new_p = []

        l_denominator = sum([sum(F_t) for F_t in F])
        for product_i in model.products:
            l_numerator = sum([F_t[product_i] for F_t in F])
            new_l.append(l_numerator / l_denominator)

            row = []
            p_denominator = sum([sum(X_t[product_i]) for X_t in X])
            for product_j in model.products:
                p_numerator = sum([X_t[product_i][product_j] for X_t in X])
                if p_denominator:
                    row.append(p_numerator / p_denominator)
                else:
                    row.append(0)
            new_p.append(row)

        model.set_lambdas(new_l)
        model.set_ros(new_p)

        return model

    def compute_psi(self, model, transaction):
        # Calculate P{Z_k (S) = 1 | F_i = 1} for all i. Uses bayes theorem
        not_offered_products = [p for p in model.products if p not in transaction.offered_products]
        A = []
        b = []
        for wanted_product in not_offered_products:
            row = []
            for transition_product in not_offered_products:
                if wanted_product == transition_product:
                    row.append(1.0)
                else:
                    row.append(-model.ro_for(wanted_product, transition_product))
            A.append(row)
            b.append(model.ro_for(wanted_product, transaction.product))

        x = list(linalg.solve(A, b)) if len(A) and len(b) else []  # Maybe all products are offered.

        psi = [0.0 if product in transaction.offered_products else x.pop(0) for product in model.products]
        psi[transaction.product] = 1.0

        return psi

    def estimate_F(self, model, transactions, psis, thetas):
        F = []
        for psi, theta, transaction in zip(psis, thetas, transactions):
            F_t = []
            for product in model.products:
                F_t.append((psi[product] * model.lambda_for(product)) / theta[transaction.product])
            F.append(F_t)
        return F

    def estimate_X(self, model, transactions, psis, thetas):
        X = []
        for psi, theta, transaction in zip(psis, thetas, transactions):
            X_t = []
            for from_product_i in model.products:
                X_t_row = []
                for to_product_j in model.products:
                    if from_product_i in transaction.offered_products:
                        X_t_row.append(0)
                    else:
                        X_t_row.append((psi[to_product_j] * model.ro_for(from_product_i, to_product_j) * theta[from_product_i]) / theta[transaction.product])
                X_t.append(X_t_row)
            X.append(X_t)
        return X
