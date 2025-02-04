from numpy import ones, array, zeros
import numpy as np
from models.__init__ import Model
from models.multinomial import MultinomiallogitModel
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_one, ZERO_LOWER_BOUND, NLP_UPPER_BOUND_INF
from estimation.optimization import Constraints
from functools import reduce


class LatentClassModel(Model):
    @classmethod
    def code(cls):
        return 'lc'
    @classmethod
    def feature(cls):
        return ['gammas',
                'multi_logit_models']

    @classmethod
    def from_data(cls, data): #data['multi_logit_models'] is a list of lists
        multi_logit_models = [MultinomiallogitModel.from_data({'products': data['products'], 'etas': multi_logit_model}) for multi_logit_model in data['multi_logit_models']]
        return cls(data['products'], data['gammas'], multi_logit_models)

    @classmethod
    def initialize_MultiType_groundtruth(cls, products, num_customer_types, folder): ## 0 represents no-purchase
        num_products = len(products)-1
        num_mnls = num_customer_types
        # generate 4 etas lists
        etas_lists = []
        for ind in range(num_customer_types):
            '''
            for i in range(int(num_mnls/num_customer_types)):
                l1 = np.random.uniform(1, 3, size=(ind*2+2)) 
                l2 = np.random.uniform(0 ,1, size=(num_products-ind*2-2))
                etas_lists.append(np.append(l1,l2))'''
            if ind == 0:
                l1 = np.exp(np.random.normal((num_customer_types-ind+num_products/num_customer_types)/num_customer_types, 1, size=(4)))
                l2 = l1.tolist()+[np.exp(-50)]*6
                etas_lists.append(l2)
            else:
                l1 = np.exp(np.random.normal((num_customer_types-ind+num_products/num_customer_types)/num_customer_types, 1, size=(2)))
                l2 = [np.exp(-50)]*(ind+1)*2+l1.tolist()+[np.exp(-50)]*(num_customer_types-ind-1)*2
                etas_lists.append(l2)
        multi_logit_models = [MultinomiallogitModel.from_data({'products': products, 'etas': etas_list}) for etas_list in etas_lists]
        # generate multi-type distribution
        gamma_lists = []
        lcmnl_models = []
        for i in range(num_customer_types):
            lists_prob = np.random.exponential(5, size=(i+1)*int(num_mnls/num_customer_types))
            lists_prob = np.append(lists_prob,np.random.exponential(1, size=num_mnls-(i+1)*int(num_mnls/num_customer_types)))
            lists_prob = lists_prob / np.sum(lists_prob)
            gamma_lists.append(lists_prob)
            lcmnl_models.append(cls(products, lists_prob.tolist(), multi_logit_models))

        #print('etas_lists:',np.array(etas_lists))
        #print('etas_lists:',np.array(gamma_lists))
        #breakpoint()
        np.save('GT/' + folder + '/GT_multi_logit_models.npy', np.array(etas_lists))
        np.save('GT/' + folder + '/GT_gammas.npy', np.array(gamma_lists))
        return lcmnl_models
        
    @classmethod
    def simple_deterministic(cls, products, amount_classes):
        gammas = generate_n_equal_numbers_that_sum_one(amount_classes)
        multi_logit_models = [MultinomiallogitModel.simple_deterministic(products) for i in range(amount_classes)]
        return cls(products, gammas, multi_logit_models)

    @classmethod
    def simple_random(cls, products, amount_classes):
        gammas = generate_n_random_numbers_that_sum_one(amount_classes)
        multi_logit_models = [MultinomiallogitModel.simple_random(products) for i in range(amount_classes)]
        return cls(products, gammas, multi_logit_models)

    def __init__(self, products, gammas, multi_logit_models):
        super(LatentClassModel, self).__init__(products)
        if len(gammas) != len(multi_logit_models):
            info = (len(gammas), len(multi_logit_models))
            raise Exception('Amount of gammas (%s) should be equal to amount of MNL models (%s).' % info)
        self.gammas = gammas
        self.multi_logit_models = multi_logit_models

    def probability_of(self, transaction):
        probability = 0.0
        for gamma, model in zip(self.gammas, self.multi_logit_models):
            probability += (gamma * model.probability_of(transaction))
        return probability

    def mnl_models(self):
        return self.multi_logit_models

    def amount_of_classes(self):
        return len(self.gammas)

    def add_new_class(self):
        percentage = 1.0 / (len(self.gammas) + 1.0)
        new_gamma = sum([gamma * percentage for gamma in self.gammas])
        self.gammas = [gamma * (1.0 - percentage) for gamma in self.gammas] + [new_gamma]
        self.multi_logit_models.append(MultinomiallogitModel.simple_deterministic(self.products))

    def add_new_class_with(self, mnl_model):
        percentage = 1.0 / (len(self.gammas) + 1.0)
        new_gamma = sum([gamma * percentage for gamma in self.gammas])
        self.gammas = [gamma * (1.0 - percentage) for gamma in self.gammas] + [new_gamma]
        self.multi_logit_models.append(mnl_model)

    def update_gammas_from(self, gammas):
        self.gammas = list(gammas)

    def parameters_vector(self):
        etas = reduce(lambda x, y: x + y, [mnl.etas for mnl in self.mnl_models()], [])
        return self.gammas + etas

    def update_parameters_from_vector(self, parameters):
        self.gammas = list(parameters)[:len(self.gammas)]
        for i, mnl_model in enumerate(self.mnl_models()):
            offset = len(self.gammas) + (i * len(mnl_model.etas))
            mnl_model.etas = list(parameters)[offset:offset + len(mnl_model.etas)]

    def constraints(self):
        return LatentClassModelConstraints(self)

    def data(self):
        return {
            'gammas': self.gammas, # list
            'multi_logit_models': [model.data()['etas'] for model in self.multi_logit_models] # list of lists
        }

    def __repr__(self):
        return '<Products: %s ; Gammas: %s >' % (self.products, self.gammas)


class LatentClassModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * NLP_UPPER_BOUND_INF

    def amount_of_constraints(self):
        return 1

    def lower_bounds_over_constraints_vector(self):
        return array([1.0])

    def upper_bounds_over_constraints_vector(self):
        return array([1.0])

    def non_zero_parameters_on_constraints_jacobian(self):
        return len(self.model.gammas)

    def constraints_evaluator(self):
        def evaluator(x):
            return array([sum(x[:len(self.model.gammas)])])
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                return (zeros(len(self.model.gammas)),
                        array(list(range(len(self.model.gammas)))))
            else:
                return ones(len(self.model.gammas))
        return jacobian_evaluator
