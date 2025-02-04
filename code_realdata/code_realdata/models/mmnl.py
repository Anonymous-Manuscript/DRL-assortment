import numpy as np
from models.__init__ import Model
from models.multinomial import MultinomiallogitModel
from GT.transactions_arrival import Transaction
from scipy.stats import truncnorm

class MixedMNLModel(Model):
    @classmethod
    def code(cls):
        return 'mmnl'
    @classmethod
    def feature(cls):
        return ['uppers', 'lowers']

    @classmethod
    def from_data(cls, data): 
        multi_logit_model = MultinomiallogitModel.simple_deterministic(data['products'])
        return cls(data['products'], data['uppers'], data['lowers'], multi_logit_model)
        
    @classmethod
    def initialize_MultiType_groundtruth(cls, products, num_customer_types, folder): ## 0 represents no-purchase
        # distribution for each etas
        num_products = len(products)-1
        uppers_list = []
        lowers_list = []
        mmnl_list = []
        for m in range(num_customer_types):
            
            uppers = np.append(np.linspace(2, 0, (m+1)*2),np.linspace(0, -20, num_products-(m+1)*2))
            lowers = np.append(np.linspace(1, 0.5, (m+1)*2),np.linspace(0.5, 0.01, num_products-(m+1)*2))
            
            uppers_list.append(uppers)
            lowers_list.append(lowers)
            multi_logit_model = MultinomiallogitModel.simple_deterministic(products)
            mmnl_list.append(cls(products, uppers, lowers, multi_logit_model))

        
        np.save('GT/' + folder + '/GT_uppers.npy', np.array(uppers_list))
        np.save('GT/' + folder + '/GT_lowers.npy', np.array(lowers_list))
        return mmnl_list
        
    def simple_deterministic(cls, products):
        num_products = len(products)-1
        uppers = np.linspace(4, 1, num_products)
        lowers = np.linspace(1, 0.1, num_products)
        multi_logit_model = MultinomiallogitModel.simple_deterministic(products)
        return cls(products, uppers, lowers, multi_logit_model)
        
    def __init__(self, products, uppers, lowers, multi_logit_model):
        super(MixedMNLModel, self).__init__(products)
        self.uppers = uppers
        self.lowers = lowers
        self.multi_logit_model = multi_logit_model
        
    def probability_distribution_over(self, offered_products):
        # sample etas
        sampled_etas = []
        for lower, upper in zip(self.lowers, self.uppers):
            
            sample = np.exp(np.random.normal(upper/4, lower))
            
            sampled_etas.append(sample)
        # update mnl
        #sampled_etas = np.array(sampled_etas)
        #sampled_etas += max(0,-min(sampled_etas)+0.01)
        self.multi_logit_model.update_parameters_from_vector(sampled_etas)
        #print(self.multi_logit_model.etas)
        #breakpoint()
        distribution = []
        for product in range(len(self.products)):
            transaction = Transaction(product, offered_products)
            distribution.append(np.min([1,self.probability_of(transaction)]))
        return distribution
    
    def probability_of(self, transaction):
        probability = self.multi_logit_model.probability_of(transaction)
        return probability

    def update_para_from(self, uppers, lowers):
        self.uppers = uppers
        self.lowers = lowers

    def data(self):
        return {
            'uppers': self.gammas, # list
            'lowers': [model.data()['etas'] for model in self.multi_logit_models] # list of lists
        }


