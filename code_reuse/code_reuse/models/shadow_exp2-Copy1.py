from models.__init__ import Model
from models.exponomial import ExponomialModel
import numpy as np
import time 
import scipy


from models import Model
from models.multinomial import MultinomiallogitModel
from utils import generate_n_equal_numbers_that_sum_one, safe_log, ZERO_LOWER_BOUND, NLP_UPPER_BOUND_INF
import numpy as np
import time 
import scipy

def generate_n_random_numbers_that_sum_one(number_n):
    den = np.random.rand(number_n) + 0.001
    sum_den = np.sum(den)
    return list(den/sum_den)

class GAM(Model):
    @classmethod
    def code(cls):
        return 'gam'
    
    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['etas'])
    
    @classmethod
    def simple_detetministic(cls, products):
        return cls(products,generate_n_random_numbers_that_sum_one(len(products)-1)*2)
    
    def __init__(self, products, etas):
        super(GAM, self).__init__(products)
        if len(etas) != len(products) * 2 - 2:
            info = (len(etas), len(products))
            raise Exception('Incorrect amount of etas (%s) for amount of products (%s)' % info)
        self.products
        self.etas = etas
        self.runtime = 0

    def etas_of(self,product):
        return 1 if product == 0 else self.etas[product - 1]
    
    def shadow_etas_of(self, product):
        return 0 if product == 0 else self.etas[len(self.products) - 1 + product - 1]

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0
        den_1 = sum([self.etas_of(product) for product in transaction.offered_products])
        den_2 = sum([self.shadow_etas_of(product) for product in self.products if product not in transaction.offered_products])
        if transaction.product != 0:
            return self.etas_of(transaction.product) / (den_1 + den_2)
        else:
            return (self.etas_of(transaction.product) + den_2) / (den_1 + den_2)
        
    def update_parameters_from_vector(self, parameters):
        self.etas = list(parameters)

    def log_probability_of(self, transaction):
        den_1 = sum([self.etas_of(product) for product in transaction.offered_products])
        den_2 = sum([self.shadow_etas_of(product) for product in self.products if product not in transaction.offered_products])
        if transaction.product != 0:
            return safe_log(self.etas_of(transaction.product)) - safe_log(den_1+den_2)
        else: 
            return safe_log(self.etas_of(transaction.product) + den_2) - safe_log(den_1+den_2)
    
    def parameters_vector(self):
        return self.etas[:len(self.etas)//2], self.etas[len(self.etas)//2:]
    
    def gradient_function(self, transation):
        gradient = np.zeros((len(self.products)-1) * 2)
        denomintor = 1
        for product in self.products[1:]:
            if product in transation.offered_products:
                denomintor += self.etas[product-1]
            else:
                denomintor += self.etas[product-1 + len(self.products) -1]
        if transation.product != 0:
            gradient[transation.product-1] += np.min([1/self.etas[transation.product-1], 1e6])
            for product in self.products[1:]:
                if product in transation.offered_products:
                    gradient[product-1] += -1/denomintor
                if product not in transation.offered_products:
                    gradient[product-1 + len(self.products)-1] += -1/denomintor
        else:
            denomintor_2 = 1
            for product in self.products[1:]:
                if product in transation.offered_products:
                    gradient[product-1] += -1/denomintor
                if product not in transation.offered_products:
                    gradient[product-1 + len(self.products)-1] += -1/denomintor
                    denomintor_2 += self.etas[product-1 + len(self.products) -1]
            for product in self.products[1:]:
                if product not in transation.offered_products:
                    gradient[product-1+len(self.products)-1] = 1/denomintor_2
        return gradient
    
    def jac_function(self, transations):
        gradient = np.zeros((len(self.products)-1) * 2)
        for transation in transations:
            gradient += self.gradient_function(transation)
        return gradient

    def estimate_from_transaction(self, products, transaction):

        base_model = MultinomiallogitModel.simple_detetministic(products)
        base_model.estimate_from_transaction(products,transaction)
        rmse_base = base_model.rmse_for(transaction)
        lowerbound = np.zeros((len(products)-1) * 2)
        upperbound = np.ones((len(products)-1) * 2) * 10000000
    
        bounds = list(zip(list(lowerbound), list(upperbound)))
        
        def constraint_func(z):
            x = z[:len(z)//2]
            y = z[len(z)//2:]
            return x - y

        constraint = {'type': 'ineq', 'fun': constraint_func}
        
        x_0 = [0] * (len(self.products) - 1) * 2
        for i in range(len(products) - 1):
            x_0[i] = base_model.etas[i]

        solve = solver(self, transaction)
        start_time = time.time()
        r = scipy.optimize.minimize(fun=solve.objective_function, x0=x_0, jac=False, bounds=bounds, constraints=constraint, method='SLSQP', options={'maxiter': 100000})
        x = r.x
        end_time = time.time()
        rmse_gam = self.rmse_for(transaction)
        if rmse_base<rmse_gam:
            self.etas = np.zeros((len(self.products)-1)*2)
            for i in range(len(base_model.etas)):
                self.etas[i] = base_model.etas[i]
        self.runtime = end_time - start_time



class solver():
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions
    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        return -self.model.log_likelihood_for(self.transactions)
    def jac_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        return - self.model.jac_function(self.transactions)
