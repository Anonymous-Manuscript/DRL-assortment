import cplex
from gurobipy import Model, GRB
from scipy.optimize import minimize
import time
from numpy import array
from utils import finite_difference, time_for_optimization

NLP_LOWER_BOUND_INF = -1e19
NLP_UPPER_BOUND_INF = 1e19
ZERO_LOWER_BOUND = 1e-6
ONE_UPPER_BOUND = 1.0 - ZERO_LOWER_BOUND
FINITE_DIFFERENCE_DELTA = 1e-7

class Settings(object):
    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            raise Exception('Must set settings for a specific estimator')
        return cls._instance

    @classmethod
    def new(cls, linear_solver_partial_time_limit,
            non_linear_solver_partial_time_limit, solver_total_time_limit):
        cls._instance = cls(linear_solver_partial_time_limit=linear_solver_partial_time_limit,
                            non_linear_solver_partial_time_limit=non_linear_solver_partial_time_limit,
                            solver_total_time_limit=solver_total_time_limit)

    def __init__(self, linear_solver_partial_time_limit,
                 non_linear_solver_partial_time_limit, solver_total_time_limit):
        self._linear_solver_partial_time_limit = linear_solver_partial_time_limit
        self._non_linear_solver_partial_time_limit = non_linear_solver_partial_time_limit
        self._solver_total_time_limit = solver_total_time_limit

    def linear_solver_partial_time_limit(self):
        return self._linear_solver_partial_time_limit

    def non_linear_solver_partial_time_limit(self):
        return self._non_linear_solver_partial_time_limit

    def solver_total_time_limit(self):
        return self._solver_total_time_limit

def set_Settings(Setting_dict):
    Settings.new(
        linear_solver_partial_time_limit=Setting_dict['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=Setting_dict['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=Setting_dict['solver_total_time_limit'],
    )
    return Settings

###### linear
class LinearProblem(object):
    def amount_of_variables(self):
        raise NotImplementedError('Subclass responsibility')

    def objective_coefficients(self):
        raise NotImplementedError('Subclass responsibility')

    def lower_bounds(self):
        raise NotImplementedError('Subclass responsibility')

    def upper_bounds(self):
        raise NotImplementedError('Subclass responsibility')

    def variable_types(self):
        raise NotImplementedError('Subclass responsibility')

    def variable_names(self):
        raise NotImplementedError('Subclass responsibility')

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')


'''class LinearSolver(object):
    def solve(self, linear_problem, profiler):
        problem = cplex.Cplex()

        problem.parameters.timelimit.set(self.cpu_time(profiler))

        problem.set_log_stream(None)
        problem.set_error_stream(None)
        problem.set_warning_stream(None)
        problem.set_results_stream(None)

        problem.objective.set_sense(problem.objective.sense.maximize)

        problem.variables.add(obj=linear_problem.objective_coefficients(),
                              lb=linear_problem.lower_bounds(),
                              ub=linear_problem.upper_bounds(),
                              types=linear_problem.variable_types(),
                              names=linear_problem.variable_names())

        problem.linear_constraints.add(lin_expr=linear_problem.constraints()['linear_expressions'],
                                       senses=''.join(linear_problem.constraints()['senses']),
                                       rhs=linear_problem.constraints()['independent_terms'],
                                       names=linear_problem.constraints()['names'])

        problem.solve()

        print('')
        print(('MIP Finished: %s' % problem.solution.get_status_string()))
        print('')

        amount_solutions = 3
        solution_pool = problem.solution.pool

        all_solutions = []
        for solution_number in range(solution_pool.get_num()):
            all_solutions.append((solution_pool.get_objective_value(solution_number), solution_number))

        final_solutions = []
        for solution_number in [x[1] for x in sorted(all_solutions, key=lambda y: y[0])[-amount_solutions:]]:
            values = {v: solution_pool.get_values(solution_number, v) for v in problem.variables.get_names()}
            final_solutions.append((problem.solution.pool.get_objective_value(solution_number), values))

        return final_solutions'''
class LinearSolver(object):
    def solve(self, linear_problem, profiler):
        # Create a new model
        model = Model()

        # Set time limit
        model.setParam('TimeLimit', self.cpu_time(profiler))

        # Set logging parameters
        model.setParam('LogToConsole', 0)

        # Assuming linear_problem.variable_types() returns a string like 'BBBB'
        raw_variable_types = linear_problem.variable_types()  # Example: 'BBBBBBBBBBBBBBBBBBBBBBBBBB'
        num_vars = len(raw_variable_types)
        # Convert the string into a list of Gurobi types
        variable_types = [GRB.BINARY] * num_vars
        # Add variables
        variables = model.addVars(
            linear_problem.variable_names(),
            obj=linear_problem.objective_coefficients(),
            lb=linear_problem.lower_bounds(),
            ub=linear_problem.upper_bounds(),
            vtype=variable_types,
            name=linear_problem.variable_names()
        )

        # Add constraints
        for i, (expr, sense, rhs, name) in enumerate(zip(
            linear_problem.constraints()['linear_expressions'],
            linear_problem.constraints()['senses'],
            linear_problem.constraints()['independent_terms'],
            linear_problem.constraints()['names']
        )):
            # Create the linear expression using Gurobi variables
            linear_expr = sum(variables[var] * coef for var, coef in zip(expr[0], expr[1]))
            if sense == 'L':
                model.addConstr(linear_expr <= rhs, name=name)
            elif sense == 'G':
                model.addConstr(linear_expr >= rhs, name=name)
            elif sense == 'E':
                model.addConstr(linear_expr == rhs, name=name)
        
        model.setObjective(
            model.getObjective(),
            GRB.MAXIMIZE  # Change to GRB.MINIMIZE if you want to minimize
        )
        
        # Optimize the model
        model.optimize()

        print('')
        print(f'MIP Finished: {model.Status}')
        print('')

        amount_solutions = 3
        all_solutions = []

        # Check if the model found any solutions
        if model.SolCount > 0:
            for solution_number in range(min(amount_solutions, model.SolCount)):
                model.setParam(GRB.Param.SolutionNumber, solution_number)
                values = {v.VarName: v.Xn for v in model.getVars()}
                objective_value = model.ObjVal
                all_solutions.append((objective_value, values))

        return all_solutions
        
    def cpu_time(self, profiler):
        return time_for_optimization(partial_time=Settings.instance().linear_solver_partial_time_limit(),
                                     total_time=Settings.instance().solver_total_time_limit(),
                                     profiler=profiler)

###### nonlinear
class NonLinearSolver(object):
    @classmethod
    def default(cls):
        return ScipySolver()

    def solve(self, non_linear_problem, profiler):
        raise NotImplemented('Subclass responsibility')

    def cpu_time(self, profiler):
        return time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                     total_time=Settings.instance().solver_total_time_limit(),
                                     profiler=profiler)


class TookTooLong(Exception):
    def __init__(self, objective_value, parameters):
        self.objective_value = objective_value
        self.parameters = parameters


class FailToOptimize(Exception):
    def __init__(self, reason):
        self.reason = reason


class ScipySolver(NonLinearSolver):
    def bounds_for(self, non_linear_problem):
        lower = list(non_linear_problem.constraints().lower_bounds_vector())
        upper = list(non_linear_problem.constraints().upper_bounds_vector())
        return list(zip(lower, upper))

    def constraints_for(self, non_linear_problem):
        lower_c = list(non_linear_problem.constraints().lower_bounds_over_constraints_vector())
        upper_c = list(non_linear_problem.constraints().upper_bounds_over_constraints_vector())
        evaluator = non_linear_problem.constraints().constraints_evaluator()

        i = 0
        constraints = []
        for l, u in zip(lower_c, upper_c):
            if l == u:
                constraints.append({'type': 'eq', 'fun': (lambda j: lambda x: evaluator(x)[j] - l)(i)})
            else:
                constraints.append({'type': 'ineq', 'fun': (lambda j: lambda x: u - evaluator(x)[j])(i)})
                constraints.append({'type': 'ineq', 'fun': (lambda j: lambda x: evaluator(x)[j] - l)(i)})
            i += 1

        return constraints

    def solve(self, non_linear_problem, profiler):
        time_limit = self.cpu_time(profiler)
        start_time = time.time()

        def iteration_callback(x):
            objective = non_linear_problem.objective_function(x)
            profiler.stop_iteration(objective)
            profiler.start_iteration()
            if time.time() - start_time > time_limit:
                raise TookTooLong(objective, x)

        bounds = self.bounds_for(non_linear_problem)
        constraints = self.constraints_for(non_linear_problem)

        profiler.start_iteration()
        try:
            r = minimize(fun=non_linear_problem.objective_function, x0=array(non_linear_problem.initial_solution()),
                         jac=False, bounds=bounds, constraints=constraints, callback=iteration_callback,
                         method='SLSQP', options={'maxiter': 100000})
            fun = r.fun
            x = r.x
            success = r.success
            status = r.status
            message = r.message
        except TookTooLong as e:
            fun = e.objective_value
            x = e.parameters
            success = True
        profiler.stop_iteration(fun)

        if not success:
            raise FailToOptimize(reason='Falla al optimizar. Estado de terminacion de scipy %s. %s' % (status, message))

        return x


class NonLinearProblem(object):
    def initial_solution(self):
        raise NotImplementedError('Subclass responsibility')

    def objective_function(self, vector):
        raise NotImplementedError('Subclass responsibility')

    def jacobian(self, vector):
        # TODO: Is it bad to define this 'finite difference' function each time jacobian is called?
        return finite_difference(self.objective_function)(vector)

    def amount_of_variables(self):
        raise NotImplementedError('Subclass responsibility')

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')


class Constraints(object):
    def lower_bounds_vector(self):
        """
            Lower bounds for parameters vector. Can be pyipopt.NLP_LOWER_BOUND_INF.
        """
        return array([])

    def upper_bounds_vector(self):
        """
            Upper bounds for parameters vector. Can be pyipopt.NLP_UPPER_BOUND_INF.
        """
        return array([])

    def amount_of_constraints(self):
        """
            Amount of constraints on model
        """
        return 0

    def lower_bounds_over_constraints_vector(self):
        """
            Lower bounds for each constraints. Can be pyipopt.NLP_LOWER_BOUND_INF.
        """
        return array([])

    def upper_bounds_over_constraints_vector(self):
        """
            Upper bounds for each constraints. Can be pyipopt.NLP_UPPER_BOUND_INF.
        """
        return array([])

    def non_zero_parameters_on_constraints_jacobian(self):
        """
            Non zero values on constraints jacobian matrix.
        """
        return 0

    def constraints_evaluator(self):
        """
            A function that evaluates constraints.
        """
        def evaluator(x):
            return 0.0
        return evaluator

    def constraints_jacobian_evaluator(self):
        """
            A function that evaluates constraints jacobian matrix.
        """
        def jacobian_evaluator(x, flag):
            if flag:
                return array([]), array([])
            else:
                return array([])
        return jacobian_evaluator
