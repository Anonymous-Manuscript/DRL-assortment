from models.__init__ import Model
import numpy as np
import time 
import scipy

class ExponomialModel(Model):
    @classmethod
    def code(cls):
        return 'exp'
    @classmethod
    def feature(cls):
        return ['utilities']

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['utilities'])

    @classmethod
    def simple_deterministic(cls, products):
        return cls(products, list(np.random.uniform(-1.0, 1.0, len(products))))
     
    @classmethod
    def instance_with_market(cls, products, market_share = 0.8):
        if market_share not in [0.2,0.5,0.8]:
            raise Exception('market_share can only be 0.2,0.5,0.8')
        u = list(np.random.uniform(-1.0, 1.0, len(products)))
        if market_share == 0.8:
            u[0] = 0.85
        elif market_share == 0.5:
            u[0] = 1.3
        elif market_share == 0.2:
            u[0] = 2.2
        return cls(products, u)
         
    @classmethod
    def instance_with_segment(cls, products, market_share = 0.8):
        if market_share not in [0.2,0.5,0.8]:
            raise Exception('market_share can only be 0.2,0.5,0.8')
        u = list(np.random.uniform(-1.0, 1.0, len(products)))
        if market_share == 0.8:
            u[0] = 0.85
        elif market_share == 0.5:
            u[0] = 1.3
        elif market_share == 0.2:
            u[0] = 2.2
        u[1:] = np.sort(u[1:])
        return cls(products, u)

    

    def __init__(self, products, utilities):
        super(ExponomialModel, self).__init__(products)
        if len(products) != len(utilities):
            info = (len(products), len(utilities))
            raise Exception('Given number of utilities (%s) does not match number of products (%s).' % info)
        self.utilities = utilities

    def utility_for(self, product):
        return self.utilities[product]

    def g(self, product, offered_products):
        better_products = [p for p in offered_products if self.utility_for(p) >= self.utility_for(product)]
        num = np.exp(-sum([self.utility_for(p) - self.utility_for(product) for p in better_products]))
        return num / len(better_products)

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0

        worse_products = [p for p in transaction.offered_products if self.utility_for(p) < self.utility_for(transaction.product)]
        worse_products = sorted(worse_products, key=lambda p: self.utility_for(p))

        accum = self.g(transaction.product, transaction.offered_products)
        for k, product in enumerate(worse_products):
            accum -= (1.0 / (len(transaction.offered_products) - k - 1.0)) * self.g(product, transaction.offered_products)
        return accum

    def parameters_vector(self):
        return self.utilities

    def update_parameters_from_vector(self, parameters):
        self.utilities = list(parameters)

    def estimate_from_transaction(self, products, transaction):
        lowerbound = - np.ones(len(products)) * 1e10
        upperbound = np.ones(len(products)) * 1e10

        bounds = list(zip(list(lowerbound), list(upperbound)))

        x_0 = self.parameters_vector()

        solve = solver(self, transaction)

        start_time = time.time()
        r = scipy.optimize.minimize(fun=solve.objective_function, x0=x_0, jac=False, bounds=bounds, method='SLSQP', options={'maxiter': 100000})
        x = r.x
        end_time = time.time()

        self.utilities = x
        self.runtime = end_time - start_time

    def data(self):
        return {
            'utilities': self.utilities, # list
        }










class solver():
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions
    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        return -self.model.log_likelihood_for(self.transactions)