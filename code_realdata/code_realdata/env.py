import numpy as np
from func import prob_of_products
import torch
import time
import random 
class market_dynamic:
    def __init__(self,args,choice_model_list,products_price):
        initial_inventory = np.array([args.ini_inv] * args.num_products)
        self.device = args.device
        self.batch_size = args.batch_size
        self.inventory_level = np.tile(initial_inventory, (self.batch_size, 1))
        self.initial_inventory = np.tile(initial_inventory, (self.batch_size, 1))
        self.total_inv = initial_inventory.sum()
        self.products_price=np.hstack((np.array([0]),products_price))
        self.choice_model_list = choice_model_list
        self.choice_model_code = choice_model_list[0].code()
        self.num_of_products=args.num_products
        self.cardinality = args.cardinality
        self.purchase=np.zeros((self.batch_size,self.num_of_products),dtype= np.int64)
        self.T=0 # remaining time
        self.arrivals = 0
    def reset(self,initial_inventory,T):
        self.inventory_level = np.tile(initial_inventory, (self.batch_size, 1))
        self.T=T
        self.arrivals = 0
    def step(self,arriving_seg,cus_feature,assortment):
        #arriving_seg: np.array([[1]]) or tensor([0])
        #assortment: torch.zeros([env.batch_size, self.num_products],dtype=torch.int)
        assert (self.inventory_level>=0).all(),'can\'t show products with no remaining inventory'
        pre = time.time()
        if self.inventory_level.any() == 0:
            self.T-=1
            return None,np.array([0])
            
        assortment = torch.from_numpy(assortment)
        arriving_seg = arriving_seg.reshape(1,)
        
        choice_model = self.choice_model_list[arriving_seg[0]]
        if self.choice_model_code == 'Assort Net': # AssortNet choice model
            ass = torch.cat((torch.tensor([[1]]), assortment), dim=1)
            probability_vector = choice_model.probability_distribution_over(ass.float())[0]
        else: #other choice model
            offered_products = (torch.nonzero(assortment[0] == 1, as_tuple=True)[0]+1).tolist()+[0]
            probability_vector = torch.tensor(choice_model.probability_distribution_over(offered_products))

        probability_vector = probability_vector / probability_vector.sum()
        # Sample an index according to the probability vector
        index = torch.multinomial(probability_vector, num_samples=self.batch_size)

        self.purchase = torch.zeros((self.batch_size, self.num_of_products+1),dtype= torch.int64)
        self.purchase.scatter_(1,index.reshape((self.batch_size,self.batch_size)),1)
        index = index.numpy()
        reward = self.products_price[index]
        
        self.inventory_level-=self.purchase[:,1:].numpy() # the first is non-purchase
        self.T-=1
        self.arrivals += 1
        
        now = time.time()
        #print('time:', now-pre)
        #breakpoint()
        return index,reward
    def get_mask(self):
        mask = self.inventory_level.copy()
        mask[self.inventory_level == 0] = 1
        mask[self.inventory_level > 0]=0
        return mask
    def all_finished(self):
        if self.T == 1:
            return True
        else:
            return False
        

class market_dynamic_feature:
    def __init__(self,args,choice_model,products_price):
        initial_inventory = np.array([args.ini_inv] * args.num_products)
        self.args = args
        self.cus_feature_indexs = []
        for m in range(4):
            self.cus_feature_indexs.append(self.args.type_cus_feature[m].shape[0])
        self.type_cus_feature = args.type_cus_feature
        self.device = args.device
        self.batch_size = args.batch_size
        self.inventory_level = np.tile(initial_inventory, (self.batch_size, 1))
        self.initial_inventory = np.tile(initial_inventory, (self.batch_size, 1))
        self.total_inv = initial_inventory.sum()
        self.products_price=np.hstack((np.array([0]),products_price))
        self.prop_features = args.prop_features
        self.choice_model = choice_model
        self.num_of_products=args.num_products
        self.cardinality = args.cardinality
        self.purchase=np.zeros((self.batch_size,self.num_of_products),dtype= np.int64)
        self.T=0 # remaining time
        self.arrivals = 0
    def reset(self,initial_inventory,T):
        self.inventory_level = np.tile(initial_inventory, (self.batch_size, 1))
        self.T=T
        self.arrivals = 0
    def step(self,arriving_seg,cus_feature,assortment):
        #arriving_seg: np.array([[1]]) or tensor([0])
        #assortment: torch.zeros([env.batch_size, self.num_products],dtype=torch.int)
        assert (self.inventory_level>=0).all(),'can\'t show products with no remaining inventory'
        pre = time.time()
        if self.inventory_level.any() == 0:
            self.T-=1
            return None,np.array([0])
                    
        assortment = torch.from_numpy(assortment)
        ass = torch.cat((torch.tensor([[1]]), assortment), dim=1)
        
        probability_vector = self.choice_model.probability_distribution_over(self.prop_features,cus_feature,ass.float())[0]

        probability_vector = probability_vector / probability_vector.sum()
        # Sample an index according to the probability vector
        index = torch.multinomial(probability_vector, num_samples=self.batch_size)

        self.purchase = torch.zeros((self.batch_size, self.num_of_products+1),dtype= torch.int64)
        self.purchase.scatter_(1,index.reshape((self.batch_size,self.batch_size)),1)
        index = index.numpy()
        reward = self.products_price[index]
        
        self.inventory_level-=self.purchase[:,1:].numpy() # the first is non-purchase
        self.T-=1
        self.arrivals += 1
        
        now = time.time()
        #print('time:', now-pre)
        #breakpoint()
        return index,reward
    def get_mask(self):
        mask = self.inventory_level.copy()
        mask[self.inventory_level == 0] = 1
        mask[self.inventory_level > 0]=0
        return mask
    def all_finished(self):
        if self.T == 1:
            return True
        else:
            return False