from models.__init__ import Model
from utils import generate_n_equal_numbers_that_sum_one, safe_log, ZERO_LOWER_BOUND, NLP_UPPER_BOUND_INF,generate_n_random_numbers_that_sum_one, generate_n_random_numbers_that_sum_m
from estimation.optimization import Constraints
import numpy as np
from numpy import ones
import time 
import scipy
import gurobipy as gp
import cvxpy as cp

class MultinomiallogitModel(Model):
    @classmethod#Class methods can be called on the class without creating an instance.
    def code(cls):
        return 'mnl'
    @classmethod
    def feature(cls):
        return ['etas']
    
    @classmethod
    def from_data(cls, data, market_share = 1):
        return cls(data['products'],data['etas'], market_share)
    
    @classmethod
    def simple_deterministic(cls, products, market_share = 1):
        return cls(products, [np.random.uniform(0, 1) for _ in range(len(products)-1)], market_share)
    
    def __init__(self, products, etas, market_share):
        super(MultinomiallogitModel, self).__init__(products)
        if len(etas) != len(products) - 1:
            info = (len(etas), len(products))
            raise Exception('Incorrect amount of etas (%s) for amount of products (%s)' % info)
        self.etas = etas
        self.market_share = market_share
        self.runtime = 0

    def utilty_of(self,product):
        return 1 if product == 0 else self.etas[product - 1]

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0
        den = sum([self.utilty_of(product) for product in transaction.offered_products])
        if transaction.product != 0:
            return self.utilty_of(transaction.product) / den * self.market_share
        else:
            fenmu = sum([self.utilty_of(product) for product in transaction.offered_products if product!=0]) * self.market_share
            return 1-fenmu/den
    def update_parameters_from_vector(self, parameters):
        self.etas = list(parameters)
        
    def constraints(self):
        return MultinomialLogitModelConstraints(self)

    def log_probability_of(self, transaction):
        return safe_log(self.probability_of(transaction))
    
    def parameters_vector(self):
        return self.etas
    
    def estimate_from_transaction(self, products, transaction):
        lowerbound = np.zeros(len(products)-1)
        upperbound = np.ones(len(products)-1) * 1e10

        bounds = list(zip(list(lowerbound), list(upperbound)))

        x_0 = self.parameters_vector()

        solve = mnl_solver(self, transaction)

        start_time = time.time()
        r = scipy.optimize.minimize(fun=solve.objective_function, x0=x_0, jac=False, bounds=bounds, method='SLSQP', options={'maxiter': 100000})
        x = r.x
        end_time = time.time()

        self.etas = x
        self.runtime = end_time - start_time
    
    def data(self):
        return {'etas': self.etas} # list


class mnl_solver():
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions
    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        return -self.model.log_likelihood_for(self.transactions)

class MultinomialLogitModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * NLP_UPPER_BOUND_INF
