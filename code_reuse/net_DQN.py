import torch.nn as nn
import torch
import numpy as np
import os
import random
from torch.distributions import Categorical
from collections import deque
from func import Cardinality_ass


def init_weights(layer):
    if type(layer) == nn.Linear:
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
        nn.init.xavier_normal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(layer.bias.data, 0)
    return layer

class QNetwork(nn.Module):
    def __init__(self, dim_in, dim_hiddens="64_32", dim_out=1):
        super(QNetwork, self).__init__()
        if dim_hiddens=="_":
            self._list = ["_"]
            self.out = nn.Linear(int(dim_in), dim_out)
        else:
            # Same architecture as Cascading DQN paper
            dim_hiddens = "{}_".format(dim_in) + dim_hiddens
            self._list = dim_hiddens.split("_")
            for i in range(len(self._list) - 1):
                _in, _out = self._list[i], self._list[i + 1]
                setattr(self, "dense{}".format(i), nn.Linear(int(_in), int(_out)).apply(init_weights))
                setattr(self, "act{}".format(i), nn.ReLU())
            self.out = nn.Linear(int(_out), dim_out)
    def forward(self, x):
        for i in range(len(self._list) - 1):
            x = getattr(self, "dense{}".format(i))(x)
            x = getattr(self, "act{}".format(i))(x)
        x = self.out(x)
        return x.view(x.shape[0], x.shape[1])  # batch_size x num_candidates

class DQN:
    def __init__(self,args,mnl_list,seed):
        self.args = args
        self.num_products = args.num_products
        self.cardinality = args.cardinality
        input_dim = args.num_products+args.cus_embedding_dim+args.t_embedding_dim
        #input_dim = args.num_products+args.num_cus_types+1
        self.MNL_V = mnl_list

        #customer type
        self.cus_embedding_dim = args.cus_embedding_dim
        self.cus_embedding = nn.Embedding(args.num_cus_types, self.cus_embedding_dim)
        #time
        self.t_embedding_dim = args.t_embedding_dim
        self.t_embedding = nn.Embedding(args.ini_inv*args.num_products*5, self.t_embedding_dim)
        
        self.QNetwork = QNetwork(input_dim,dim_hiddens=args.layer, dim_out=args.num_products).to(args.device)
        
        self.target_model = QNetwork(input_dim,dim_hiddens=args.layer, dim_out=args.num_products).to(args.device)
        self.update_target_model()
        
        self.update_step = 100
        self.counter = 0
        
        # init replay memory
        self.replayMemory = deque()
        self.replaySize = 0
        self.sample_batch_size = 4#args.batch_size
        self.loss_func=nn.MSELoss()

        self.epsilon = 1
        self.EPSILON_MIN = 0.01
        self.epsilon_decay = args.epsilon_decay
        self.device = args.device
        self.dir = args.stream
        
        torch.manual_seed(args.net_seed+seed)
        np.random.seed(args.net_seed+seed)
        random.seed(args.net_seed+seed)
    def select_action(self, mask,inputs,arriving_seg,train_):
        #mask:np.array([[0,0,0,0,0,0,0,0,0,0]])
        avail_prod = np.ones(self.num_products)*(1-mask[0])#一维
        if np.random.rand() <= self.epsilon and bool(train_):#随机选几个东西展示
            avail_num = self.num_products-len(mask[0].nonzero()[0])
            if avail_num>self.cardinality:
                choose_num = random.randint(1,self.cardinality)#inclusive inclusive
            else:
                choose_num = random.randint(1,avail_num)#inclusive inclusive
            ass_ = np.random.choice(avail_prod.nonzero()[0], choose_num, replace=False)#index
            ass = np.zeros((1,self.num_products))
            ass[:,ass_] = 1#onehot
        else:#根据Q值作为price，定assortment，mask的产品price为负无穷
            act_values = self.QNetwork(torch.DoubleTensor(inputs).to(self.device))
            mask_q_values = act_values.cpu().detach().numpy()[0]
            mask_q_values[mask[0].nonzero()[0]] = -1e5
            ass = Cardinality_ass(self.MNL_V[arriving_seg[0]],mask_q_values,self.cardinality)
            ass = ass.reshape((1,-1))
        return ass  # assortment的onehot表示

    def remember(self, state, action, reward, next_state, done):
        '''put the sequence into replay memory'''
        if self.replaySize > 5000:
            self.replayMemory.popleft()
            self.replaySize -= 1
        self.replayMemory.append(
            [state, action, reward, next_state, done])
        self.replaySize += 1

    def update(self,optimizer,scheduler):
        #从replaymemory中sample样本出来训练
        # step1: obtain random minibatch from replay memory
        
        minibatch = random.sample(self.replayMemory, self.sample_batch_size)  # 从replayMemory中sample大小为batchSize的样本出来
        state_batch = torch.stack([torch.flatten(data[0].float())
                    for data in minibatch])#.cuda(2)  # dim: [batchSize个array],每个array都是10*5shape的
        action_batch = np.array([data[1] for data in minibatch]).astype(int)#.cuda(2)  # dim: [[actionlistLen]*batchSize]
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])
                                        ).float().to(self.device) # dim: [batchSize个]
        nextState_batch = torch.stack([torch.flatten(data[3].float()
                                                     ) for data in minibatch])#.cuda(2)  # dim: [[multPerInput*stateDim]*batchSize]
        dones = torch.from_numpy(np.array([data[4] for data in minibatch])
                                        ).float().to(self.device) # dim: [batchSize个]
        # step2: 分别calculate Q Value and y
        QTValue_batch = []
        act_values = []

        
        transfered_state_batch = state_batch[:,:self.num_products]
        transfered_state_batch = torch.cat((transfered_state_batch, self.t_embedding(state_batch[:,self.num_products].int())),dim=1)
        transfered_state_batch = torch.cat((transfered_state_batch, self.cus_embedding(state_batch[:,-1].int())),dim=1)
        
        transfered_nextState_batch = nextState_batch[:,:self.num_products]
        transfered_nextState_batch = torch.cat((transfered_nextState_batch, self.t_embedding(nextState_batch[:,self.num_products].int())),dim=1)
        transfered_nextState_batch = torch.cat((transfered_nextState_batch, self.cus_embedding(nextState_batch[:,-1].int())),dim=1)
        
        
        for b in range(self.sample_batch_size):
            with torch.no_grad():
                
                QTValue_batch_b = self.target_model.forward(transfered_nextState_batch)#.unsqueeze(0).double().to(self.device)
                
                mask_q_values = QTValue_batch_b.cpu().numpy()[0]
                mask_q_values[nextState_batch[b,:self.num_products] == 0] = -1e5
                V = self.MNL_V[nextState_batch[b,-1].int()]
                try:
                    T_assort = Cardinality_ass(V,mask_q_values,self.cardinality)
                except:
                    breakpoint()
                    T_assort = Cardinality_ass(V,mask_q_values,self.cardinality)
                QTValue_batch.append(torch.sum(QTValue_batch_b*(V*T_assort)/(1+(V@T_assort))))

            
            act_values_b = self.QNetwork(transfered_state_batch)#.unsqueeze(0).double().to(self.device)
            
            
            V = self.MNL_V[state_batch[b,-1].int()]
            act_values.append(torch.sum(act_values_b[0]*torch.tensor(V*action_batch[b,:])/(1+(V@action_batch[b,:]))))
            
        QTValue_batch = torch.stack(QTValue_batch,0).unsqueeze(1).to(self.device)
        act_values = torch.stack(act_values,0).unsqueeze(1).to(self.device)
        y_batch = reward_batch + (1 - dones.reshape(self.sample_batch_size,-1))*QTValue_batch
        loss = self.loss_func(act_values,y_batch)
        # step3:梯度下降
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        if optimizer.param_groups[0]['lr']>self.args.lr_min:
            scheduler.step()
        
        if self.epsilon>self.EPSILON_MIN:
            self.epsilon *= self.epsilon_decay
        # update target network
        self.counter += 1
        if self.counter % self.update_step == 0:
            self.update_target_model()
    '''
    def best_batch_action(self,state_batch):
        state_batch = state_batch.numpy()
        #从state_batch中提取出mask(表示对应的state，哪些商品不能选了)
        action_index = np.zeros((self.args.cardinality,self.sample_batch_size)).astype(int)
        ass = torch.zeros((self.sample_batch_size,self.args.num_products))#选的定为1
        mask = state_batch[:,:5].copy()
        mask[mask>0] = 1#可选的,batchsize*num_products
        #根据state和mask选最好的
        act_values = self.QNetwork(state_batch.double().to(self.device))
        torch.argmax(act_values,axis=0)
        return action_index.T#array'''
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.QNetwork.state_dict())
        #print('***** Target network updated *****')
    
    def save_model(self, args):
        print("Save the agent.")
        torch.save(self.QNetwork.state_dict(),self.dir+'QNet.pkl')

    def load_weights(self, args):
        print("Load the agent: {}QNet.pkl".format(self.dir))
        self.QNetwork.load_state_dict(torch.load(self.dir+'QNet.pkl'))
        
        
        
        
if __name__ == '__main__':
    print("=== _test ===")
    dim_in = 10
    x = torch.randn(3, dim_in, device="cpu")
    model = QNetwork(dim_in=dim_in,
                     dim_hiddens="256_32").to("cpu")
    breakpoint()
    out = model(x)
    print(out)


