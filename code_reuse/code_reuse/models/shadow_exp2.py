from models.__init__ import Model
from models.exponomial import ExponomialModel
import numpy as np
import time 
import scipy


class S_EXP2(Model):
    @classmethod
    def code(cls):
        return 'S-exp2'
    @classmethod
    def feature(cls):
        return ['utilities']
    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['utilities'])

    @classmethod
    def simple_deterministic(cls, products):
        a = list(np.random.uniform(-1.0, 1.0, len(products)))
        b = [-100] * len(products) 
        return cls(products, a, b)


    def __init__(self, products, utilities, shadow_utilities):
        self.products = products
        self.utilities = utilities
        self.shadow_u = shadow_utilities

    def utility_for(self, product):
        return self.utilities[product]
    
    def shadow_u_for(self, product):
        return self.shadow_u[product]

    def g(self, product, offered_products):
        utility = np.zeros(len(self.products))
        for p in self.products:
            if p in offered_products:
                utility[p] = self.utility_for(p)
            else:
                utility[p] = self.shadow_u_for(p)
        better_products = [p for p in self.products if utility[p] >= utility[product]]
        num = np.exp(-sum([utility[p] - utility[product] for p in better_products]))
        
        return num / len(better_products)

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0
        
        utility = np.zeros(len(self.products))
        for p in self.products:
            if p in transaction.offered_products:
                utility[p] = self.utility_for(p) 
            else:
                utility[p] = self.shadow_u_for(p)

        if transaction.product != 0:

            worse_products = [p for p in self.products if utility[p] < utility[transaction.product]]
            worse_products = sorted(worse_products, key=lambda p: utility[p])

            accum = self.g(transaction.product, transaction.offered_products)
            for k, product in enumerate(worse_products):
                accum -= (1.0 / (len(self.products) - k - 1.0)) * self.g(product, transaction.offered_products)
            return accum
        else:
            purchase_prop = 0
            for prod in self.products:
                if prod not in transaction.offered_products[1:]:
                    worse_products = [p for p in self.products if utility[p] < utility[prod]]
                    worse_products = sorted(worse_products, key=lambda p: utility[p])
                    accum = self.g(prod, transaction.offered_products)
                    for k, product in enumerate(worse_products):
                        accum -= (1.0 / (len(self.products) - k - 1.0)) * self.g(product, transaction.offered_products)
                    purchase_prop += accum
            return purchase_prop

    def parameters_vector(self):
        return self.utilities, self.shadow_u

    def update_parameters_from_vector(self, parameters):
        self.utilities = list(parameters)
    

    def estimate_from_transaction(self, products, transaction):
        base_model = ExponomialModel.simple_deterministic(products)
        base_model.estimate_from_transaction(products,transaction)
        rmse_base = base_model.rmse_for(transaction)
        lowerbound = - np.ones(len(products)*2) * 1e10
        upperbound = np.ones(len(products)*2) * 1e10
        upperbound[len(products)] = -1e9

        bounds = list(zip(list(lowerbound), list(upperbound)))

        def constraint_func(z):
            x = z[:len(z)//2]
            y = z[len(z)//2:]
            return x - y
        
        constraint = {'type': 'ineq', 'fun': constraint_func}
        a = list(np.random.uniform(-1.0, 1.0, len(products)))     
        x_0 = a + [i-0.01 for i in a]
        solve = solver(self, transaction)

        r = scipy.optimize.minimize(fun=solve.objective_function, x0=x_0, jac=False, bounds=bounds, constraints=constraint,method='SLSQP', options={'maxiter': 10000})
        x = r.x
     

        self.utilities = x[:len(x)//2]
        self.shadow_u = x[len(x)//2:]
        rmse_shadow = self.rmse_for(transaction)
        if rmse_base<rmse_shadow:
            self.utilities = base_model.utilities
            self.shadow_u = [-10000] * len(self.products)
    def data(self):
        return {
            'utilities': self.utilities, # list
        }
    
class solver():
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions
    def objective_function(self, parameters):
        self.model.utilities = parameters[:len(parameters)//2]
        self.model.shadow_u = parameters[len(parameters)//2:]
        #return self.model.rmse_for(self.transactions)
        return -self.model.log_likelihood_for(self.transactions) 