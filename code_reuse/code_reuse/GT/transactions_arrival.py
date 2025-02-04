from itertools import chain, combinations
import random
import numpy as np
from func import NpEncoder
import json

class Transaction(object):
    @classmethod
    def from_json(cls, json_list):
        return [cls(d['product'], d['offered_products']) for d in json_list]
    
    @classmethod
    def all_for(cls, model):
        products = set(model.products) - {0}
        for offer_set in powerset(products):
            for product in [0] + sorted(offer_set):
                yield cls(product, [0] + sorted(offer_set))
    
    def __init__(self, product, offered_products):
        self.product = product
        self.offered_products = offered_products

    def as_json(self):
        return {'product': self.product, 'offered_products': self.offered_products}

class TransactionGenerator(object):
    def __init__(self, model):
        self.model = model # could be a list of models

    def gene_MultiType_rl_data(self, args):
        ranked_lists = self.model[0].ranked_lists

        N = len(ranked_lists[0])-1
        M = len(self.model)

        dict_ = {}
        # generate arrival data of customer types
        k = args.k
        interval = (args.L - 1) / (M+1)
        time_range = []
        for i in range(M):
            point = round(interval + i * interval)
            time_range.append(point)
        time_range = sorted(set(time_range))
        time_range = np.array(time_range)
        num_periods = args.T
        mean_len = args.L
        for i in range(num_periods):  # sequence data
            T = np.random.randint(mean_len-int(args.L/10), mean_len+int(args.L/10))
            input_sequence = []
            for t in range(T):
                arriving_seg = np.exp(-k * np.absolute(t - time_range))
                arriving_seg = arriving_seg / arriving_seg.sum()
                arriving_seg = np.random.choice(a=np.arange(M),
                                                p=arriving_seg)
                input_sequence.append(arriving_seg)
            dict_[i] = input_sequence
        # generate transaction for each customer type
        custype_transdata_onehot = {}
        transactions = {}
        for j in range(M):
            custype_transdata_onehot[str(j)] = []
            transactions[str(j)] = []
        for sequence in list(dict_.values()):
            for arriving_seg in sequence:
                ass = random.sample(list(range(1,1+N)), 5)+[0]
                trans = self.generate_transaction_for_type(arriving_seg, ass)
                transactions[str(arriving_seg)].append(trans)
                ass_onehot = np.zeros(N + 1)
                ass_onehot[ass] = 1
                custype_transdata_onehot[str(arriving_seg)].append(
                    np.append(ass_onehot, trans.product))  # assortment onehot 表示,以及选的index

        return custype_transdata_onehot, transactions, dict_

    def gene_lcmnl_data(self, args): #self.model is lcmnl
        N = args.num_products
        M = len(self.model)

        dict_ = {}
        # generate arrival data of customer types
        k = args.k
        interval = (100 - 1) / (M+1)
        time_range = []
        for i in range(M):
            point = round(interval + i * interval)
            time_range.append(point)
        time_range = sorted(set(time_range))
        time_range = np.array(time_range)
        
        num_periods = 500
        mean_len = 100
        
        for i in range(num_periods):  # sequence data
            T = np.random.randint(mean_len-10, mean_len+10)
            input_sequence = []
            for t in range(T):
                arriving_seg = np.exp(-k * np.absolute(t - time_range))
                arriving_seg = arriving_seg / arriving_seg.sum()
                arriving_seg = np.random.choice(a=np.arange(M),
                                                p=arriving_seg)
                input_sequence.append(arriving_seg)
            dict_[i] = input_sequence
        # generate transaction for each customer type
        custype_transdata_onehot = {}
        transactions = {}
        for j in range(M):
            custype_transdata_onehot[str(j)] = []
            transactions[str(j)] = []
        for sequence in list(dict_.values()):
            for arriving_seg in sequence:
                ass = random.sample(list(range(1,1+N)), 5)+[0]
                trans = self.generate_transaction_for_type(arriving_seg, ass)
                transactions[str(arriving_seg)].append(trans)
                ass_onehot = np.zeros(N + 1)
                ass_onehot[ass] = 1
                custype_transdata_onehot[str(arriving_seg)].append(
                    np.append(ass_onehot, trans.product))  # assortment onehot 表示,以及选的index

        return custype_transdata_onehot, transactions, dict_

    def gene_mmnl_data(self, args): #self.model is mmnl
        N = args.num_products
        M = len(self.model)

        dict_ = {}
        # generate arrival data of customer types
        k = args.k
        interval = (100 - 1) / (M+1)
        time_range = []
        for i in range(M):
            point = round(interval + i * interval)
            time_range.append(point)
        time_range = sorted(set(time_range))
        time_range = np.array(time_range)
        
        num_periods = 500
        mean_len = 100
        
        for i in range(num_periods):  # sequence data
            T = np.random.randint(mean_len-10, mean_len+10)
            input_sequence = []
            for t in range(T):
                arriving_seg = np.exp(-k * np.absolute(t - time_range))
                arriving_seg = arriving_seg / arriving_seg.sum()
                arriving_seg = np.random.choice(a=np.arange(M),
                                                p=arriving_seg)
                input_sequence.append(arriving_seg)
            dict_[i] = input_sequence
        # generate transaction for each customer type
        custype_transdata_onehot = {}
        transactions = {}
        for j in range(M):
            custype_transdata_onehot[str(j)] = []
            transactions[str(j)] = []
        for sequence in list(dict_.values()):
            for arriving_seg in sequence:
                ass = random.sample(list(range(1,1+N)), 5)+[0]
                trans = self.generate_transaction_for_type(arriving_seg, ass)
                transactions[str(arriving_seg)].append(trans)
                ass_onehot = np.zeros(N + 1)
                ass_onehot[ass] = 1
                custype_transdata_onehot[str(arriving_seg)].append(
                    np.append(ass_onehot, trans.product))  # assortment onehot 表示,以及选的index

        return custype_transdata_onehot, transactions, dict_

    def generate_transaction_for_type(self, cus_type, offered_products):
        distribution = self.model[cus_type].probability_distribution_over(offered_products)
        try:
            a = np.random.multinomial(1, distribution, 1)
            purchased_product = list(a[0]).index(1)
        except Exception as e:
            print("发生了异常：", e)
            print(distribution)
        return Transaction(purchased_product, offered_products) # purchased_product: 0,1,...,N
        
    def generate_for(self, lists_of_offered_products):
        transactions = []
        for i, offered_products in enumerate(lists_of_offered_products):
            transactions.append(self.generate_transaction_for(offered_products))
        return transactions

    def generate_transaction_for(self, offered_products):
        distribution = self.model.probability_distribution_over(offered_products)
        try:
            a = np.random.multinomial(1, distribution, 1)
            purchased_product = list(a[0]).index(1)
        except Exception as e:
            print("发生了异常：", e)
            print(distribution)
        return Transaction(purchased_product, offered_products) # purchased_product: 0,1,...,N
    
class TransactionGenerator2(object):
    def __init__(self, model):
        self.model = model

    def generate_for(self, lists_of_offered_products_1, lists_of_offered_products_2):
        transactions = []
        for i in range(len(lists_of_offered_products_1)):
            transactions.append(self.generate_transaction_for(lists_of_offered_products_1[i],lists_of_offered_products_2[i]))
        return transactions

    def generate_transaction_for(self, offered_products_1, offered_products_2):
        distribution = self.model.probability_distribution_over(offered_products_1, offered_products_2)
        try:
            a = np.random.multinomial(1, distribution, 1)
            purchased_product = list(a[0]).index(1)

        except Exception as e:
            print("发生了异常：", e)
            print(distribution)
        return Transaction(purchased_product, offered_products_1)
    
class OfferedProductsGenerator(object):
    def __init__(self, products):
        self.products = products

    def generate_distinct(self, amount, min_times_offered=1, max_times_offered=1):
        offer_sets = []
        while len(offer_sets) < amount:
            offered_products = self.generate_offered_products()
            if offered_products != [0] and offered_products not in offer_sets:
                amount_of_times_offered = random.choice(list(range(min_times_offered, max_times_offered + 1)))
                for i in range(amount_of_times_offered):
                    offer_sets.append(offered_products)
        random.shuffle(offer_sets)
        return offer_sets


    
    def generate(self, amount, prop =0.5):
        offer_sets = []
        while len(offer_sets) < amount:
            offer_sets.append(self.generate_offered_products_2(prop))
        return offer_sets
    
    def generate_offered_products_2(self, prop):
        offered_products = []
        for i in range(1,len(self.products)):
            a = np.random.uniform(0,1)
            if a < prop:
                offered_products.append(i)
        offered_products = [0] + offered_products
        return offered_products



    def generate_random_size(self, amount):
        offer_sets = []
        while len(offer_sets) < amount:
            size = np.random.randint(3,7)
            offer_sets.append(self.generate_offered_products(size))
        return offer_sets
    
    def generate_diversity(self, amount, diversity):
        offer_sets = []
        while len(offer_sets) < amount:
            size = 4
            a = self.generate_offered_products(size)
            for j in range(diversity):
                offer_sets.append(a)
        return offer_sets


    def generate_with_size(self, amount, minimum, maximum):
        offer_sets = []
        while len(offer_sets) < amount:
            a = self.generate_offer_set_with_size(minimum, maximum)
            offer_sets.append(a)
        return offer_sets

    def generate_offer_set_with_size(self, minimum, maximum):
        offered = {0}
        size = random.choice(list(range(minimum, maximum + 1)))
        while len(offered) < size + 1:
            offered.add(random.choice(self.products))
        return sorted(offered)

    def generate_offered_products(self, size = 5):
        a = np.random.choice(range(1,11),size=size,replace=False)
        a = np.sort(a)
        a = [i for i in a]
        offered_products = [0] + a
        return offered_products
    
    def generate_all_products(self, amount):
        offer_sets =[]
        while len(offer_sets) < amount:
            offer_sets.append(self.generate_offer_all())
        return offer_sets

    def generate_one_product(self, amount):
        offer_sets = []
        while len(offer_sets) < amount:
            offer_sets.append(self.generate_offer_one())
        return offer_sets


    def generate_offer_all(self):
        return self.products
    
    def generate_offer_one(self):
        offered = {0}
        offered.add(random.choice(self.products))
        return sorted(offered)
    



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))