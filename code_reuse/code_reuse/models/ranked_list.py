from models.__init__ import Model
from utils import generate_n_equal_numbers_that_sum_one
import numpy as np
import random

def generate_n_random_numbers_that_sum_one(number_n):
    den = np.random.rand(number_n) + 0.001
    sum_den = np.sum(den)
    return list(den/sum_den)

class RankedListModel(Model):
    @classmethod
    def code(cls):
        return 'rl'
    @classmethod
    def feature(cls):
        return ['betas',
                'ranked_lists']
    
    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['ranked_lists'], data['betas'])
    @classmethod
    def initialize_MultiType_groundtruth(cls, products, num_customer_types, folder): ## 0 represents no-purchase
        # no-purchase ranks 1 in the first list
        # in half of the lists, small products rank high; in another half, large products rank high
        # for 5 lists, and 2 customer types, the betas are like [[0.1,0.2,0.7,0,0],[0.1,0.2,0.2,0.3,0.2]]
        # when saving, the 0.1 is abandoned
        num_products = len(products)-1

        if num_products==10:
            numbers = list(range(1, 9))
            full_numbers = set(range(1, 11))
            pairs = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]
    
            rank_list0 = [list(range(0, num_products + 1))]  # no-purchase ranks 1
            all_lists = rank_list0
            num_lists = 20
            for pair_i in range(4):
                for round in range(5):
                    remaining_numbers = list(full_numbers - set(pairs[pair_i]))
                    list_1 = np.random.permutation(pairs[pair_i])
                    list_2 = np.random.permutation(remaining_numbers)
                    list_ = list(np.append(list_1,list_2))
                    list_.insert(random.randint(2*pair_i+2, num_products), 0)
                    all_lists.append(list_)
        elif num_products==20:
            numbers = list(range(1, 17))
            full_numbers = set(range(1, 21))
            pairs = [(numbers[i], numbers[i + 1], numbers[i + 2], numbers[i + 3]) for i in range(0, len(numbers) - 1, 4)]
    
            rank_list0 = [list(range(0, num_products + 1))]  # no-purchase ranks 1
            all_lists = rank_list0
            num_lists = 20
            for pair_i in range(4):
                for round in range(5):
                    remaining_numbers = list(full_numbers - set(pairs[pair_i]))
                    list_1 = np.random.permutation(pairs[pair_i])
                    list_2 = np.random.permutation(remaining_numbers)
                    list_ = list(np.append(list_1,list_2))
                    list_.insert(random.randint(4*pair_i+4, num_products), 0)
                    all_lists.append(list_)
        
        betas = []
        rlModels = []
        for i in range(num_customer_types):
            size_ = int((i + 1) * (num_lists / num_customer_types))
            lists_prob = np.random.exponential(1, size=size_)
            lists_prob = lists_prob / np.sum(lists_prob)
            lists_prob = lists_prob * 0.9
            #lists_prob = np.append(0.1, lists_prob)
            lists_prob = np.pad(lists_prob, (0, num_lists - size_),
                                'constant', constant_values=(0, 0))
            rlModels.append(cls(products, all_lists, lists_prob))
            betas.append(lists_prob) #
        betas = np.array(betas)

        np.save('GT/' + folder + '/GT_ranked_lists.npy', all_lists)
        np.save('GT/' + folder + '/GT_cus_types.npy', betas)

        return rlModels

    @classmethod
    def simple_deterministic(cls, products, ranked_lists):
        betas = generate_n_equal_numbers_that_sum_one(len(ranked_lists))[1:]
        return cls(products, ranked_lists, betas)    
    
    @classmethod
    def simple_deterministic_independent(cls, products):
        ranked_lists = [[i] + sorted(set(products) - {i}) for i in range(len(products))]
        return cls.simple_deterministic(products, ranked_lists)
    
    def __init__(self, products, ranked_lists, betas):
        super(RankedListModel, self).__init__(products)
        if len(betas) + 1 != len(ranked_lists):
            info = (len(betas), len(ranked_lists))
            raise Exception('Amount of betas (%s) should be one less than of ranked lists (%s).' % info)
        if any([len(ranked_list) != len(products) for ranked_list in ranked_lists]):
            info = (products, ranked_lists)
            raise Exception('All ranked list should have all products.\n Products: %s\n Ranked lists: %s\n' % info)

        self.ranked_lists = ranked_lists #list of lists
        self.betas = betas # np.array shaping [num_customer_types, len(ranked_lists)-1]

    def compatibility_matrix_for(self, transactions):
        matrix = []
        for t in transactions:
            matrix.append([1.0 if self.are_compatible(r, t) else 0.0 for r in self.ranked_lists])
        return np.array(matrix)

    def probabilities_for(self, transactions):
        return np.dot(self.compatibility_matrix_for(transactions), np.array(self.all_betas()))

    def probability_of(self, transaction):
        probability = 0
        for ranked_list_number, ranked_list in self.ranked_lists_compatible_with(transaction):
            probability += self.beta_for(ranked_list_number)
        return np.min([1,probability])

    def amount_of_ranked_lists(self):
        return len(self.ranked_lists)

    def all_betas(self):
        return [self.beta_for(ranked_list_number) for ranked_list_number in range(len(self.ranked_lists))]

    def beta_for(self, ranked_list_number):
        return 1 - sum(self.betas) if ranked_list_number == 0 else self.betas[ranked_list_number - 1]

    def set_betas(self, all_betas):
        self.betas = all_betas[1:]

    def are_compatible(self, ranked_list, transaction):
        if transaction.product not in ranked_list:
            return False
        better_products = ranked_list[:ranked_list.index(transaction.product)]
        return all([p not in transaction.offered_products for p in better_products])

    def ranked_lists_compatible_with(self, transaction):
        if transaction.product not in transaction.offered_products:
            return []

        compatible_ranked_lists = []
        for i, ranked_list in enumerate(self.ranked_lists):
            if self.are_compatible(ranked_list, transaction):
                compatible_ranked_lists.append((i, ranked_list))
        return compatible_ranked_lists

    def add_ranked_list(self, ranked_list):
        if ranked_list not in self.ranked_lists:
            percentage = 1.0 / (len(self.betas) + 2.0)
            new_beta = sum([beta * percentage for beta in self.all_betas()])
            self.betas = [beta * (1.0 - percentage) for beta in self.betas] + [new_beta]
            self.ranked_lists.append(ranked_list)

    def parameters_vector(self):
        return self.betas

    def update_parameters_from_vector(self, parameters):
        self.betas = list(parameters)

    def data(self):
        return {
            'betas': self.betas, # list
            'ranked_lists': self.ranked_lists, # list of lists
        }