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
        
        self.num_cus_types = args.num_cus_types
        self.ini_inv = initial_inventory[0]
        self.reuse_type = args.reuse_type
        self.on_road_hours = np.zeros((self.batch_size,self.num_of_products,self.ini_inv),dtype= np.int64)
        self.history_length = 40
        self.purchase_history = torch.zeros((self.num_cus_types,self.history_length,self.num_of_products),dtype= torch.int64)
        self.cus_rate = np.array([0.2,0.25,0.3,0.35])
        self.prod_rate = np.linspace(0.2,0.4,10)
    def reset(self,initial_inventory,T):
        self.inventory_level = np.tile(initial_inventory, (self.batch_size, 1))
        self.T=T
        self.arrivals = 0
        
        self.on_road_hours = np.zeros((self.batch_size,self.num_of_products,self.ini_inv),dtype= np.int64)
        self.purchase_history = torch.zeros((self.num_cus_types,self.history_length,self.num_of_products),dtype= torch.int64)
    def step(self,arriving_seg,assortment):
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
        index = index.numpy() # 0 is no-purchase
        reward = self.products_price[index]
        
        self.on_road_hours -= 1
        self.on_road_hours[self.on_road_hours < 0] = 0
        if reward[0] != 0:#otherwise inventory unchange
            lend_index = np.argmin(self.on_road_hours,axis=2)[0,index[0]-1]
            if self.reuse_type=='prod':
                use_time = np.random.negative_binomial(2,self.prod_rate[index[0]-1])+1
            elif self.reuse_type=='cus':
                use_time = np.random.negative_binomial(2,self.cus_rate[int(arriving_seg[0])])+1
            else:
                use_time = np.random.negative_binomial(2,0.25)+1
            self.on_road_hours[0,index[0]-1,lend_index] = use_time
        self.inventory_level = np.sum(np.where(self.on_road_hours>0,0,1),axis=2)

        cat_ = torch.zeros((self.num_cus_types, 1, self.num_of_products),dtype= torch.int64)
        self.purchase_history = torch.cat( (cat_,self.purchase_history[:,:-1,:]) , dim=1)
        #try:
            #self.purchase_history[arriving_seg[0][0],0,:] = self.purchase[0,:-1]
        #except:
        self.purchase_history[arriving_seg[0],0,:] = self.purchase[0,:-1]
        
        #self.inventory_level-=self.purchase[:,1:].numpy() # the first is non-purchase
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
        
