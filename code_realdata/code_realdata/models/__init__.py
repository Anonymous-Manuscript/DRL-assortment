from GT.transactions_arrival import Transaction
from utils import safe_log
from math import sqrt
import numpy as np
class Model(object):
    """
        Reprsents a mathmatical model for Discrete Choice Consumer Decision
    """
    def __init__(self, products):
        if products != list(range(len(products))):
            raise Exception('Products should be enteredas an order consecutive list.')
        self.products = products

    @classmethod
    def code(cls):
        raise NotImplementedError('Subclass responsibility')
    
    @classmethod
    def from_data(cls, data):
        for klass in cls.__subclasses__():
            if data['code'] == klass.code():
                return klass.from_data(data)
        raise Exception('No model can be created from data')
    
    @classmethod
    def simple_detetministic(cls, *arg, **kwargs):
        """
            must return a default model with simple pdf parameters to use as an initial solution for estimators.
        """
        raise NotImplementedError('Subclass responsibility')
    
    def probability_of(self, transaction):
        """
            Must return the probability of a transaction
        """
        raise NotImplementedError('Subclass responsibility')
    def assortment(self, prices):
        """
            reture the opt assortment
        """
        raise NotImplementedError('Subclass responsibility')

    def revenue_of_a_assortment(self, prices, assortment):
        revenue = 0
        for product in assortment[1:]:
            transaction = Transaction(product, assortment)
            revenue += self.probability_of(transaction) * prices[product - 1]
        return revenue


    def probability_distribution_over(self, offered_products):
        distribution = []
        for product in range(len(self.products)):
            transaction = Transaction(product, offered_products)
            distribution.append(np.min([1,self.probability_of(transaction)]))
        return distribution

    def log_probability_of(self, transaction):
        return safe_log(self.probability_of(transaction))
    
    def log_likelihood_for(self, transactions):
        result = 0
        cache = {}
        for transaction in transactions:
            cache_code = (transaction.product, tuple(transaction.offered_products))
            if cache_code in cache:
                log_probability = cache[cache_code]
            else:
                log_probability = self.log_probability_of(transaction)
                cache[cache_code] = log_probability
            result += log_probability
        return result / len(transactions)

    def revenue_know_ground(self, ground_model, prices, assortment):
        if len(prices) != len(self.products) - 1:
            raise Exception('Incorrect length of prices')
        revenue = 0
        for product in assortment:
            if product != 0:
                transaction = Transaction(product, assortment)
                revenue += prices[product-1] * ground_model.probability_of(transaction)
        return revenue
    

    def soft_rmse_for(self, ground_model):
        rmse = 0.0
        amount_terms = 0.0
        for t in Transaction.all_for(self):
            rmse += ((self.probability_of(t) - ground_model.probability_of(t)) ** 2)
            amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def soft_mape_for(self, ground_model):
        rmse = 0.0
        amount_terms = 0.0
        for t in Transaction.all_for(self):
            if t.product != 0:
                rmse += ((self.probability_of(t) - ground_model.probability_of(t)) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))
    
    def rmse_for(self, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability = self.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability - float(product == transaction.product)) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def rmse_known_ground(self, ground_model, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability_1 - probability_2) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))
    

