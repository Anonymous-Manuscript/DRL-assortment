import numpy as np
import random
from func import Cardinality_ass,prob_of_products,exp_rev
from arg import init_parser
import scipy.stats as st
import cvxpy as cp
import json

class OA_agent:
    def __init__(self,args,env_,products_price):
        self.market=env_
        self.batch_size = args.batch_size
        self.cardinality = args.cardinality
        self.products_price=products_price
        self.total_reward=np.zeros((self.batch_size,1))
        self.N=len(products_price)

        self.detail = args.detail
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((args.num_products+1,110))
        self.purchase_matrix = np.zeros((args.num_products+1,110))
        self.i = 0
    def OA(self):
        # random
        ass = np.zeros((self.batch_size,self.N))
        range_ = self.market.inventory_level.nonzero()[1]
        for i in range(self.batch_size):
            try:
                random_choose = random.sample(list(range_), random.randint(1,self.cardinality))
            except:
                random_choose = random.sample(list(range_), random.randint(1,len(range_)))
            ass[i][random_choose] = 1
        ass = ass*self.market.inventory_level
        ass[ass>0] = 1
        return ass
    def reset(self,initial_inventory,T):
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((self.N+1,110))
        self.purchase_matrix = np.zeros((self.N+1,110))
        self.i = 0
    def step(self,arriving_seg):
        OA_ass = self.OA()
        index_,reward = self.market.step(arriving_seg, OA_ass)
        self.total_reward += reward
        if self.detail:
            self.change_of_inv_list.append(self.market.inventory_level.mean(0))
            self.change_of_R_list.append(self.total_reward.mean(0)[0])
            self.offer_matrix[OA_ass[0].nonzero()[0],self.i] += 1
            self.purchase_matrix[index_,self.i] += 1
            self.i += 1
        

class myopic_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.market = env_
        self.batch_size = args.batch_size
        self.MNL_para = MNL_para
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality=args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)

        self.detail = args.detail
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((args.num_products+1,110))
        self.purchase_matrix = np.zeros((args.num_products+1,110))
        self.i = 0
    def myopic_ass(self,arriving_seg):
        myopic_ass = []
        for i,cus in enumerate(arriving_seg):
            V = self.MNL_para[cus[0]]
            myopic_ass.append(
                Cardinality_ass(V,self.products_price[i],self.cardinality))
            #for item in np.array(myopic_ass).nonzero()[1]:
                #if self.products_price[0,item] == 0:
                    #breakpoint()
        return np.array(myopic_ass)
    def reset(self,initial_inventory,T,products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((self.N+1,110))
        self.purchase_matrix = np.zeros((self.N+1,110))
        self.i = 0
    def step(self,arriving_seg):
        #self.exam(arriving_seg)
        myopic_ass = self.myopic_ass(arriving_seg)
        choose_index,reward = self.market.step(arriving_seg, myopic_ass)
        if (self.market.inventory_level).sum() == 0:
            choose_index = 0
        self.total_reward += reward
        copy_inv = self.market.inventory_level.copy()
        copy_inv[copy_inv>0] = 1
        self.products_price = self.products_price*copy_inv#不摆库存为0的

        if self.detail:
            self.change_of_inv_list.append(self.market.inventory_level.mean(0))
            self.change_of_R_list.append(self.total_reward.mean(0)[0])
            self.offer_matrix[myopic_ass[0].nonzero()[0],self.i] += 1
            
            self.purchase_matrix[choose_index,self.i] += 1
            
            self.i += 1

def E_penalty_function(x):
    return (1-np.exp(-x))*(np.e/(np.e-1))
def L_penalty_function(x):
    return x

class E_IB_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.market = env_
        self.batch_size = args.batch_size
        self.initial_inventory = env_.initial_inventory
        self.MNL_para = MNL_para
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)

        self.detail = args.detail
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((args.num_products+1,110))
        self.purchase_matrix = np.zeros((args.num_products+1,110))
        self.i = 0
    def IB_ass(self,arriving_seg):
        IB_ass = []
        for i,cus in enumerate(arriving_seg):
            V = self.MNL_para[cus[0]]
            r_ = E_penalty_function(self.market.inventory_level[i]/
                                    self.initial_inventory[i]) \
                                    * self.products_price[i]
            IB_ass.append(Cardinality_ass(V,r_,self.cardinality))
            #print('IB_ass:',r_,np.array(IB_ass).nonzero()[1])
        return np.array(IB_ass)
    def reset(self, initial_inventory, T):
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((self.N+1,110))
        self.purchase_matrix = np.zeros((self.N+1,110))
        self.i = 0
    def step(self,arriving_seg):
        IB_ass = self.IB_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, IB_ass)
        self.total_reward += reward
        #print('IB_ass_reward:',self.total_reward)
        if (self.market.inventory_level).sum() == 0:
            choose_index = 0

        if self.detail:
            self.change_of_inv_list.append(self.market.inventory_level.mean(0))
            self.change_of_R_list.append(self.total_reward.mean(0)[0])
            self.offer_matrix[IB_ass[0].nonzero()[0],self.i] += 1
            self.purchase_matrix[choose_index,self.i] += 1
            self.i += 1

class sub_t_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.args = args
        self.market = env_
        self.batch_size = args.batch_size
        self.initial_inventory = env_.initial_inventory
        self.MNL_para = MNL_para
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
        self.t = 0
        self.segprob = args.seg_prob
    def sub_t_ass(self,arriving_seg):
        sub_t_ass = []
        for i,cus in enumerate(arriving_seg):
            T = 109
            lambda_ = 0.91
            d_t,alpha_t = information_t(self.MNL_para,self.market.inventory_level[0],self.segprob)
            d_t = lambda_*d_t
            D_t = variable_Dt(d_t,self.t,T)
            D_tS = variable_DtS(D_t,alpha_t,self.market.inventory_level[0])
            delta = []
            for i in range(self.N):
                p1 = len(D_tS[i][np.where(D_tS[i]>self.market.inventory_level[0][i])])/10000
                j_list = list(range(self.N))
                j_list.remove(i)
                sum_ = 0
                for ind,j in enumerate(j_list):
                    p2 = len(D_tS[j][np.where(D_tS[j]<=self.market.inventory_level[0][j])])/10000
                    p3 = len(D_t[i][np.where(D_t[i]>self.market.inventory_level[0][i])])/10000
                    sum_ += alpha_t[i][ind]*p2*p3
                delta.append(self.products_price[0][i]*(p1-sum_))
            delta = np.array(delta)
            prices = np.array(self.products_price[0])-delta
            copy_inv = self.market.inventory_level[0].copy()
            copy_inv[copy_inv>0] = 1
            prices = prices*copy_inv#不摆库存为0的
            
            V = self.MNL_para[cus[0]]
            sub_t_ass.append(Cardinality_ass(V,prices,self.cardinality))
            #print('sub_t_ass:',prices,np.array(sub_t_ass).nonzero()[1])
        if random.random()>lambda_ and self.t<109:
            self.t += 1 
        return np.array(sub_t_ass)
    def reset(self, initial_inventory, T, products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        self.t = 0
    def step(self,arriving_seg):
        sub_t_ass = self.sub_t_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, sub_t_ass)
        self.total_reward += reward   
        #print('sub_t_reward:',self.total_reward)
        if (self.market.inventory_level).sum() == 0:
            choose_index = 0
        if self.t<109:
            self.t += 1   



class DP_Greedy_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.args = args
        self.market = env_
        self.batch_size = args.batch_size
        self.initial_inventory = env_.initial_inventory
        self.MNL_para = MNL_para
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
        self.t = 0
        self.segprob = args.seg_prob
        
        self.M = args.num_cus_types
        self.T = 110
        self.v_hat = np.zeros((self.T+1,self.N))#theta in the paper
        self.A_t = np.zeros((self.T,self.M,self.N))
        self.build()

        self.detail = args.detail
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((args.num_products+1,110))
        self.purchase_matrix = np.zeros((args.num_products+1,110))
        self.i = 0
    def build(self):
        for t in range(self.T-1,-1,-1):
            temp_sum = 0
            for j in range(self.M):
                A_tj = Cardinality_ass(self.MNL_para[j],self.products_price[0]-self.v_hat[t+1],self.cardinality)
                self.A_t[t,j,:] = A_tj
                temp_sum += self.segprob[j]*prob_of_products(A_tj, self.MNL_para[j])[1:]
            for i in range(self.N):
                self.v_hat[t,i] = self.v_hat[t+1,i] + temp_sum[i]*(self.products_price[0,i]-self.v_hat[t+1,i])/self.initial_inventory[0,i]
    def DP_Greedy_ass(self,arriving_seg):
        DP_Greedy_ass = []
        for i,cus in enumerate(arriving_seg):
            prices = np.array(self.products_price[0])-self.v_hat[self.t+1,:]
            copy_inv = self.market.inventory_level[0].copy()
            copy_inv[copy_inv>0] = 1
            prices = prices*copy_inv#不摆库存为0的
            DP_Greedy_ass.append(Cardinality_ass(self.MNL_para[cus[0]],prices,self.cardinality))
            #print(prices)
            #print(DP_Greedy_ass)
            #for item in np.array(DP_Greedy_ass).nonzero()[1]:
                #if prices[item] <= 0:
                    #breakpoint()
        return np.array(DP_Greedy_ass)
    def reset(self, initial_inventory, T, products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        self.t = 0
        
        self.change_of_inv_list=[]
        self.change_of_R_list=[]
        self.offer_matrix = np.zeros((self.N+1,110))
        self.purchase_matrix = np.zeros((self.N+1,110))
        self.i = 0
    def step(self,arriving_seg):
        DP_Greedy_ass = self.DP_Greedy_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, DP_Greedy_ass)
        if (self.market.inventory_level).sum() == 0:
            choose_index = 0
            
        self.total_reward += reward
        if self.t<self.T-1:
            self.t += 1        
            
        if self.detail:
            self.change_of_inv_list.append(self.market.inventory_level.mean(0))
            self.change_of_R_list.append(self.total_reward.mean(0)[0])
            self.offer_matrix[DP_Greedy_ass[0].nonzero()[0],self.i] += 1
            self.purchase_matrix[choose_index,self.i] += 1
            self.i += 1       
            
class DP_Rollout_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.args = args
        self.market = env_
        self.batch_size = args.batch_size
        self.initial_inventory = env_.initial_inventory
        self.MNL_para = MNL_para
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
        self.t = 0
        self.segprob = args.seg_prob
        
        self.M = args.num_cus_types
        self.T = 110
        self.v_hat = np.zeros((self.T+1,self.N))#theta in the paper
        self.A_t = np.zeros((self.T,self.M,self.N))
        self.value = np.zeros((self.T+1,self.N,self.initial_inventory[0,0]+1))# V(q) in the paper
        self.build()
    def build(self):
        for t in range(self.T-1,-1,-1):
            temp_sum = 0
            for j in range(self.M):
                A_tj = Cardinality_ass(self.MNL_para[j],self.products_price[0]-self.v_hat[t+1],self.cardinality)
                self.A_t[t,j,:] = A_tj
                temp_sum += self.segprob[j]*prob_of_products(A_tj, self.MNL_para[j])[1:]
            for i in range(self.N):
                self.v_hat[t,i] = self.v_hat[t+1,i] + temp_sum[i]*(self.products_price[0,i]-self.v_hat[t+1,i])/self.initial_inventory[0,i]
                for q in range(1,self.initial_inventory[0,0]+1):
                    self.value[t,i,q] = self.value[t+1,i,q] + temp_sum[i]\
                                       *(self.products_price[0,i]-(self.value[t+1,i,q]-self.value[t+1,i,q-1]))
    def DP_Rollout_ass(self,arriving_seg):
        DP_Greedy_ass = []
        for i,cus in enumerate(arriving_seg):
            prices = np.array(self.products_price[0])
            for i in range(self.N):
                if self.market.inventory_level[0,i] > 0:
                    prices[i] = prices[i]-(self.value[self.t+1,i,self.market.inventory_level[0,i]]
                                           -self.value[self.t+1,i,self.market.inventory_level[0,i]-1])
            copy_inv = self.market.inventory_level[0].copy()
            copy_inv[copy_inv>0] = 1
            prices = prices*copy_inv#不摆库存为0的
            DP_Greedy_ass.append(Cardinality_ass(self.MNL_para[cus[0]],prices,self.cardinality))
        return np.array(DP_Greedy_ass)
    def reset(self, initial_inventory, T, products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        self.t = 0
    def step(self,arriving_seg):
        DP_Greedy_ass = self.DP_Rollout_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, DP_Greedy_ass)
        if (self.market.inventory_level).sum() == 0:
            choose_index = 0
            
        self.total_reward += reward   
        #print('sub_t_reward:',self.total_reward)
        if self.t<self.T-1:
            self.t += 1               




              

        
from itertools import combinations

def information_t(MNL_para,inventory_level,seg_prob):
    MNL_para_t = MNL_para.copy()
    MNL_para_t[:,np.where(inventory_level==0)] = 0
    V_sum = (MNL_para_t.sum(1)+1).reshape((len(seg_prob),1))
    p_i = MNL_para/V_sum
    d_it = (p_i*seg_prob.reshape((len(seg_prob),1))).sum(0)
    alpha_ijt = 0
    for i in range(len(seg_prob)):
        mnl = np.repeat(MNL_para[i].reshape((1,len(inventory_level))),len(inventory_level),axis=0)
        mnl_t = np.repeat(MNL_para_t[i].reshape((1,len(inventory_level))),len(inventory_level),axis=0)
        row, col = np.diag_indices_from(mnl)
        mnl[row,col] = 0
        row, col = np.diag_indices_from(mnl_t)
        mnl_t[row,col] = 0
        alpha_ijt += mnl/((mnl_t.sum(1)+1).reshape((len(inventory_level),1)))*seg_prob[i]
    
    return d_it,alpha_ijt
    
def variable_Dt(d_t,t,T):  
    D_t=[]
    for i in range(len(d_t)):
        D_t.append(st.poisson.rvs(d_t[i]*(T-t), loc=0, size=10000))
    return np.array(D_t)

def variable_DtS(D_t,alpha_t,inventory_level):
    minus = D_t-inventory_level.reshape((len(inventory_level),1))
    minus[minus<0] = 0
    minus[inventory_level==0,:]=0
    N = len(inventory_level)
    D_tS = D_t + (alpha_t.reshape((N,N,1))*np.repeat(minus.reshape((1,N,10000)),N,axis=0)).sum(0)
    return D_tS


def primal(SS,seg_prob_,products_price,MNL_para,inventory_level):
    breakpoint()
    num_var = len(SS[0])+len(SS[1])+len(SS[2])+len(SS[3])
    vars_ = cp.Variable(num_var)
    v = 0
    constraints=[]
    Rev=cp.Constant(0)
    num = 0
    for z,S in zip(range(4),SS):
        constraints.append(cp.sum(vars_[num:num+len(S)])<=seg_prob_[z])
        for s in S:
            Rev += seg_prob_[z]*exp_rev(np.array(s),MNL_para[z],products_price)*vars_[v]
            v += 1
        num += len(S)
    for i in range(10):
        num = 0
        sum_ = 0
        for z,S in zip(range(4),SS):
            prob_list = []
            for s in S:
                prob_list.append(seg_prob_[z]*prob_of_products(np.array(s),MNL_para[z])[i])
            sum_ += prob_list@vars_[num:num+len(S)]
            num += len(S)
        constraints.append(sum_<=inventory_level[i])
    constraints.append(0 <= vars_)
    objective=cp.Maximize(Rev)
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    beta = list(list(prob.solution.primal_vars.values())[0])  
    primal1 = np.array(beta[:len(SS[0])])
    primal1[primal1<0] = 0
    primal2 = np.array(beta[len(SS[0]):len(SS[0])+len(SS[1])])
    primal2[primal2<0] = 0
    primal3 = np.array(beta[len(SS[0])+len(SS[1]):len(SS[0])+len(SS[1])+len(SS[2])])
    primal3[primal3<0] = 0
    primal4 = np.array(beta[len(SS[0])+len(SS[1])+len(SS[2]):])
    primal4[primal4<0] = 0
    primal = [primal1,primal2,primal3,primal4]
    return primal


def dual(SS,seg_prob_,products_price,MNL_para,inventory_level):
    breakpoint()
    vars_ = cp.Variable(14)
    obj = cp.Constant(0)
    constraints=[]
    obj += vars_[:10]@inventory_level
    obj += vars_[10:]@seg_prob_
    for z,S in zip(range(4),SS):
        for s in S:
            left = seg_prob_[z]*(vars_[:10]@prob_of_products(np.array(s),MNL_para[z])[:-1])
            right = seg_prob_[z]*(products_price@prob_of_products(np.array(s),MNL_para[z])[:-1])
            constraints.append(right <= vars_[z+10]+left)
    constraints.append(0 <= vars_)
    objective=cp.Minimize(obj)
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    beta = list(list(prob.solution.primal_vars.values())[0])  
    return beta

def column_gene(products_price,beta_dual,SS,MNL_para,seg_prob_):
    breakpoint()
    r_ = products_price-beta_dual[:10]
    obj_max = 0
    add = True
    for z in range(4):
        ass_ = Cardinality_ass(MNL_para[z],r_,4)
        obj_ = seg_prob_[z]*exp_rev(ass_,MNL_para[z],r_)-beta_dual[z+10]
        if obj_ > obj_max:
            obj_max = obj_
            z_max = z
            ass_max = ass_
    if obj_max == 0:
        add = False
    else:
        SS[z_max].append(list(ass_max))
    return SS,add

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    





