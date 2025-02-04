from itertools import chain, combinations
import random
import numpy as np
from func import NpEncoder
import json

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

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


class Transaction_with_cus_feature(object):
    def __init__(self, product, offered_products, cus_feature):
        self.product = product
        self.offered_products = offered_products
        self.cus_feature = cus_feature

def data_for_estimtion(csv_data,prop_features):
    #create：product feature, customer feature, assortment01
    id_list = list(np.unique(csv_data['srch_id']))
    random.shuffle(id_list)
    train_srch_list = id_list[0: int(len(id_list) * 0.8)]
    test_srch_list = id_list[int(len(id_list) * 0.8):]

    train_data = read(train_srch_list,csv_data,prop_features)
    test_data = read(train_srch_list,csv_data,prop_features)

    print('data tranformed!')
    return train_data,test_data
    
def read(srch_list,csv_data,prop_features):
    num_prods = len(np.unique(csv_data['prop_id']))
    num_prods_features = 6
    num_cus_features = 6
    num_cus_types = 4
    transactions = []
    transactions_with_cus_feature = []
    type_transactions = {}
    sample_list = np.zeros((1, num_prods + 1, num_prods_features + num_cus_features + num_prods + 1))
    choose_list = []
    transdata_onehot = []
    custype_transdata_onehot = {}
    for j in range(num_cus_types):
        type_transactions[str(j)] = []
        custype_transdata_onehot[str(j)] = []
    i = 0
    for srch in srch_list:
        srch_data = csv_data[csv_data['srch_id'] == srch]
        cus_type = np.nonzero(srch_data.iloc[0, -4:].values)[0][0]
        cus_feature = srch_data.iloc[0, 9:15].values

        show_prods = srch_data.loc[:, 'prop_id'].values + 1
        show_prods = np.append(0, show_prods)
        if srch_data['booking_bool'].sum() == 0:
            choose = 0
        else:
            choose = srch_data[srch_data['booking_bool'] == 1]['prop_id'].values[0] + 1
        transactions.append(Transaction(choose, show_prods))
        transactions_with_cus_feature.append(Transaction_with_cus_feature(choose, show_prods,cus_feature))
        type_transactions[str(cus_type)].append(Transaction(choose, show_prods))

        ass = np.zeros(num_prods + 1)
        ass[show_prods] = 1
        transdata_onehot.append(np.append(ass, choose))
        custype_transdata_onehot[str(cus_type)].append(np.append(ass, choose))
        ass = np.repeat(ass[np.newaxis, :], num_prods + 1, 0)

        multi = np.zeros((num_prods + 1, num_prods_features))
        multi[show_prods] = 1
        prop_fea = prop_features * multi
        cus_fea = np.repeat(cus_feature[np.newaxis, :], num_prods + 1, 0)
        sample = np.concatenate((np.concatenate((prop_fea, cus_fea), 1), ass), 1)
        sample_list = np.concatenate((sample_list, sample[np.newaxis, :, :]), 0)
        choose_list.append(choose)


        i += 1

    sample_list = sample_list[1:]
    choose_list = np.array(choose_list)

    return transactions,transactions_with_cus_feature,type_transactions,sample_list,choose_list,transdata_onehot,custype_transdata_onehot











    