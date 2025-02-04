import numpy as np
import gurobipy as gp
from gurobipy import *
from itertools import combinations
import scipy.sparse as sp
from GT.transactions_arrival import Transaction
import copy

def solve_lp(n,cardinality,GT_choice_model_list,prices,inventory):#rank_list,cus_type,
    GT_list_prob = [] # each element is, a list if prob sparse matrix with different cardinalities, for each cus type
    #rank_list_ = rank_list-1
    #rank_list_[rank_list_==-1] = n
    for GT in GT_choice_model_list:
        assort_list = []
        num_of_S = 0
        prob_card_list = []
        for i in range(1,cardinality+1):
            cat_list_i = list(map(list, combinations(list(np.arange(1, n + 1)), i)))
            num_of_S += len(cat_list_i)
            cat_list = np.array(cat_list_i)
            cat_list = np.hstack((np.zeros((len(cat_list_i), 1), dtype=cat_list.dtype), cat_list)).tolist()

            prob = []
            for assort in cat_list:
                offered_prod_prob = []
                for prod in assort:
                    offered_prod_prob.append(GT.probability_of(Transaction(prod,assort)))
                prob.append(offered_prod_prob)#prob list for each offered product, including 0

                onehot_assort = np.zeros(n+1)
                onehot_assort[assort] = 1
                assort_list.append(onehot_assort)
            
            prob = np.array(prob).reshape([1, -1]).tolist()[0]

            ind1 = np.tile(np.arange(len(cat_list)), (i+1, 1)).T.reshape(1, -1).tolist()[0]
            ind2 = np.array(cat_list).reshape([1, -1]).tolist()[0]
            assert len(prob)==len(ind1)
            
            prob_card_list.append(sp.csc_matrix(sp.csc_array((prob, (ind1, ind2)), shape=(len(cat_list), n+1))))
            #print(prob_card_list)
            #breakpoint()
        GT_list_prob.append(prob_card_list)

    prices = np.insert(prices, 0, 0)
    def solve_current(arrival_types):
        T = len(arrival_types)
        m = gp.Model('model')
        y = m.addMVar(shape=((1, num_of_S * T)), lb=0.0, vtype=GRB.CONTINUOUS, name='y')
        r_matrix = sp.vstack([
                       sp.vstack([GT_list_prob[arrival_types[t]][card].multiply(prices).sum(axis=1)
                                  for card in range(cardinality)])
                       for t in range(T)]).A.flatten()
        obj = y @ (r_matrix)
        m.setObjective(obj, GRB.MAXIMIZE)

        for prod in range(1,n+1):
            c1 = m.addConstr(((y @ sp.vstack([
                                        sp.vstack([GT_list_prob[arrival_types[t]][card].getcol(prod)
                                                   for card in range(cardinality)])
                                        for t in range(T)]).A.flatten())
                               <= inventory[prod-1])
                              , name='c1')
        '''
        c2 = m.addConstrs((quicksum( y[0,S * T + t] for S in range(num_of_S))
                           <= 1 for t in range(T))
                          , name='c2')'''
        c2 = m.addConstrs((quicksum( y[0,t * num_of_S + S] for S in range(num_of_S))
                           <= 1 for t in range(T))
                          , name='c2')

        m.optimize()
        UB = m.objVal
        y_S_T = np.array(y.x.tolist()[0])#prob of showing assortment 1,2,3,... in time 1; prob of showing assortment 1,2,3,... in time 2; ...
        return UB,y_S_T
    return solve_current,assort_list
