from models.__init__ import Model
from utils import generate_n_equal_numbers_that_sum_one
import numpy as np
import random

def generate_n_random_numbers_that_sum_one(number_n):
    den = np.random.rand(number_n) + 0.001
    sum_den = np.sum(den)
    return list(den/sum_den)

class GSPModel(Model):
    @classmethod
    def code(cls):
        return 'gsp'
    @classmethod
    def feature(cls):
        return ['betas',
                'ranked_lists',
                'k_probs',
                'k_list']
    
    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['ranked_lists'], data['betas'], data['k_probs'], data['k_list'])
    @classmethod
    def initialize_MultiType_groundtruth(cls, products, num_customer_types, folder): ## 0 represents no-purchase
        num_lists = 15
        #rank lists: rank of products, including 0
        ranked_lists = [list(np.arange(len(products)))]
        for l in range(num_lists-1):
            shuffled_products = products.copy()
            random.shuffle(shuffled_products)
            ranked_lists.append(shuffled_products)
        #k list
        k_list = [1,2]
        #k prob: prob of picking a k
        k_probs = generate_n_random_numbers_that_sum_one(len(k_list))
        #betas: prob of picking a list
        gspModels = []
        betas = []
        for i in range(num_customer_types):
            beta_list = [0.1]
            betas_ = list(np.array(generate_n_random_numbers_that_sum_one(num_lists-1))*0.9)
            beta_list = beta_list + betas_

            gspModels.append(cls(products, ranked_lists, beta_list[1:], k_probs, k_list))
            betas.append(beta_list)

        np.save('GT/' + folder + '/GT_ranked_lists.npy', np.array(ranked_lists))
        np.save('GT/' + folder + '/GT_k_probs.npy', np.array(k_probs))
        np.save('GT/' + folder + '/GT_k_list.npy', np.array(k_list))
        np.save('GT/' + folder + '/GT_cus_types.npy', np.array(betas)[:,1:])

        return gspModels
    
    def __init__(self, products, ranked_lists, betas, k_probs, k_list):
        super(GSPModel, self).__init__(products)
        self.ranked_lists = ranked_lists #list of lists
        self.betas = betas # np.array shaping [num_customer_types, len(ranked_lists)]
        self.k_probs = k_probs
        self.k_list = k_list

    def probability_of(self, transaction):
        #used in directUB.py and self.probability_distribution_over(self, offered_products)
        probability = 0
        if transaction.product not in transaction.offered_products:
            return 0
        max_k = np.max(self.k_list)#k_list:[1,2] choose the first or second
        for i, ranked_list in enumerate(self.ranked_lists):
            count_k = self.compatible_k(ranked_list, transaction)
            if count_k > max_k-1:
                pass
            else:
                probability += self.beta_for(i)*self.k_probs[count_k]
        return np.min([1,probability])

    def beta_for(self, ranked_list_number):
        return 1 - sum(self.betas) if ranked_list_number == 0 else self.betas[ranked_list_number - 1]

    def compatible_k(self, ranked_list, transaction):
        if transaction.product not in ranked_list:
            return False
        better_products = ranked_list[:ranked_list.index(transaction.product)]
        #how many better_products are in offered_products
        count_k = 0
        for p in better_products:
            if p in transaction.offered_products:
                count_k += 1
        # count_k = 1 means there is one product that is offered and better than transaction.product
        return count_k

    def data(self):
        return {
            'betas': self.betas, # list
            'ranked_lists': self.ranked_lists, # list of lists
            'k_probs': self.k_probs, # list
            'k_list': self.k_list, # list of lists
        }