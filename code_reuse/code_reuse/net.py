import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
import time

def E_penalty_function(x):
    return (1-np.exp(-x))*(np.e/(np.e-1))
    
class A2C(nn.Module):
    def __init__(self, args):
        super(A2C, self).__init__()

        self.time_filter = nn.Parameter(torch.DoubleTensor(args.num_products,40*args.num_cus_types).uniform_(0.18, 0.22))
        
        self.MLP = args.MLP
        self.decode_type = None
        self.num_products = args.num_products
        self.num_cus_types = args.num_cus_types
        self.cardinality = args.cardinality
        self.args = args
        self.device = args.device
        
        #customer type
        self.cus_embedding_dim = args.cus_embedding_dim
        self.cus_embedding = nn.Embedding(args.num_cus_types, self.cus_embedding_dim)
        #time
        self.t_embedding_dim = args.t_embedding_dim
        self.t_embedding = nn.Embedding(args.ini_inv*args.num_products*5, self.t_embedding_dim)

        share = []
        share.append(nn.Linear(2*args.num_products+self.cus_embedding_dim+self.t_embedding_dim, args.w[0]))#
        share.append(nn.ReLU())
        for h in range(args.h-1):
            share.append(nn.Linear(args.w[h], args.w[h+1]))
            share.append(nn.ReLU())
        share.append(nn.Linear(args.w[args.h-1], args.nn_out))
        share.append(nn.ReLU())
        self.share = nn.Sequential(*share)
        
        self.critic = nn.Sequential( nn.Linear(args.nn_out, 1) )   

        if self.MLP:
            self.actor = nn.Sequential(
                 *[nn.Linear(args.nn_out, args.num_products)]#,nn.ReLU(),nn.Linear(args.num_products, args.num_products)]
                   #,nn.ReLU(),nn.Linear(args.num_products, args.num_products)]
            )
        else:
            self.actor = nn.RNN(args.nn_out, args.num_products+1,1)   ###########
        
    def forward(self, history, x):
        time_filter = nn.functional.leaky_relu( torch.sum( history.view(-1,self.num_products).T.to(self.device)*self.time_filter,dim=1), negative_slope=0.2)
        
        x = self.share( torch.cat((time_filter.reshape((1,self.num_products)),x), dim=1) )
        value = self.critic(x)
        if not self.MLP:
            return x, value
        else:
            score = self.actor(x)##########
            return score, value##########
    def roll_out(self, env, input_sequence, i):
        ass_log_softmax_list_ = []
        values = []
        R = []
        m_dones=[]
        mean_entropy = 0
        s = 0
        for num_step in range(self.args.num_steps):
            arriving_seg = torch.tensor(input_sequence[:, i]).to(self.device)
            cus_type_embedding = self.cus_embedding(arriving_seg)
            s = torch.tensor(env.inventory_level/env.initial_inventory).to(self.device)
            #s = s*self.t_embedding(torch.tensor(env.arrivals).to(self.device))
            s = torch.cat((s,self.t_embedding(torch.tensor(env.arrivals).to(self.device).unsqueeze(0))), dim=1)
            #s = torch.cat((s,torch.tensor([[env.arrivals/env.total_inv]]).to(self.device)), dim=1)
            s = torch.cat((s,cus_type_embedding), dim=1)
            # 关键语句
            x, value = self.forward(env.purchase_history,s)
            #########
            mask = torch.from_numpy(env.get_mask())
            # Select the indices of the next nodes in the sequences
            if self.MLP:
                assortment, entropy, ass_log_softmax = self._select_node_MLP(env,
                                x, mask.bool())  # Squeeze out steps dimension
            else:
                assortment, entropy, ass_log_softmax = self._select_node_RNN(env,
                                x, mask.bool())  # Squeeze out steps dimension
            mean_entropy += entropy
            _, reward = env.step(arriving_seg,assortment.numpy())
            #要输出的东西
            ass_log_softmax_list_.append(ass_log_softmax)
            values.append(value)
            m_dones.append(1 - env.all_finished())
            R.append(torch.DoubleTensor([reward]))
            i += 1
            if num_step == self.args.num_steps-1 or env.all_finished():#这个roll out结束
                if env.all_finished():
                    next_value = 0#done之后这个值是没用的
                    break
                arriving_seg = torch.randint(1, self.num_cus_types, (1,)).to(self.device)
                cus_type_embedding = self.cus_embedding(arriving_seg)
                next_state = torch.tensor(env.inventory_level/env.initial_inventory).to(self.device)
                #next_state = next_state*self.t_embedding(torch.tensor(env.arrivals).to(self.device))
                next_state = torch.cat((next_state,self.t_embedding(torch.tensor(env.arrivals).to(self.device).unsqueeze(0))), dim=1)
                #next_state = torch.cat((next_state,torch.tensor([[env.arrivals/env.total_inv]]).to(self.device)), dim=1)
                next_state = torch.cat((next_state,cus_type_embedding), dim=1)
                _, next_value = self.forward(env.purchase_history,next_state)
                break
        # Collected lists, return Tensor
        return (
            torch.stack(ass_log_softmax_list_, 1),#shape(batch_size,T)
            torch.cat(values,1),
            torch.cat(R,1).to(self.device),###
            mean_entropy.double(),
            torch.tensor(m_dones).to(self.device),
            i,
            next_value
        )
            

    def calc_entropy(self, _log_p):
        entropy = -(_log_p * _log_p.exp()).sum(2).sum(1).mean()
        return entropy
    
    def _select_node_MLP(self, env, score, mask):
        score[mask] = -1e20
        p = torch.log_softmax(score, dim=1)
        dist = Categorical(p)
        entropy = dist.entropy().mean()#想让 entropy 变大，更加随机
        ass = torch.zeros([env.batch_size, self.num_products],dtype=torch.int)
        if self.decode_type == "greedy":
            _, idx1 = torch.sort(p, descending=True)  # descending为False，升序，为True，降序
            selected = idx1[:,:self.cardinality]
            ass.scatter_(1,selected,1)
            ass = ass*torch.logical_not(mask)#注意：不是mask着的东西才能摆
            ass_log_softmax = (ass*p).sum(axis=1)#是一个长度为batch_size的张量
        elif self.decode_type == "sampling":
            selected = p.exp().multinomial(self.cardinality,replacement=True)#放回的抽样
            ass.scatter_(1,selected,1)
            ass = ass*torch.logical_not(mask)#注意：不是mask着的东西才能摆
            ass_log_softmax = (ass*p).sum(axis=1)
        else:
            assert False, "Unknown decode type"
        return ass, entropy ,ass_log_softmax
    
    def _select_node_RNN(self, env, x, mask):
        assortment = torch.zeros([env.batch_size, self.num_products+1],dtype=torch.int).to(self.device)
        entropy = 0
        ass_log_softmax = 0
        mask = torch.cat((mask, torch.tensor([[1]])), -1).to(self.device)
        if self.decode_type == "greedy":
            scores,hidden_layer = self.actor(x.repeat(self.args.cardinality,1).reshape(self.args.cardinality,1,-1))
            for node_num in range(self.args.cardinality):
                score = scores[node_num]
                score = score.masked_fill(mask.bool(), value=torch.tensor(-1e20))
                p = torch.log_softmax(score, dim=1)
                dist = Categorical(p)
                entropy += dist.entropy().mean()#想让 entropy 变大，更加随机

                _, idx1 = torch.sort(p, descending=True)  # descending为False，升序，为True，降序
                selected = idx1[:,:1]
                
                ass = torch.zeros([env.batch_size, self.num_products+1],dtype=torch.int).to(self.device).scatter(1,selected,1)
                ass_log_softmax += (ass*p).sum(axis=1)#是一个长度为batch_size的张量
                if selected == self.num_products:
                    break
                assortment += ass
                mask[:,selected[0][0]] = 1
                mask[:,-1] = 0
            #ass = ass*torch.logical_not(mask)#注意：不是mask着的东西才能摆
                
        elif self.decode_type == "sampling":
            scores,hidden_layer = self.actor(x.repeat(self.args.cardinality,1).reshape(self.args.cardinality,1,-1))
            for node_num in range(self.args.cardinality):
                score = scores[node_num]
                score = score.masked_fill(mask.bool(), value=torch.tensor(-1e20))
                p = torch.log_softmax(score, dim=1)
                dist = Categorical(p)
                entropy += dist.entropy().mean()#想让 entropy 变大，更加随机
                selected = p.exp().multinomial(1)
                ass = torch.zeros([env.batch_size, self.num_products+1],dtype=torch.int).to(self.device).scatter(1,selected,1)
                ass_log_softmax += (ass*p).sum(axis=1)#是一个长度为batch_size的张量
                if selected == self.num_products:
                    break
                assortment += ass
                mask[:,selected[0][0]] = 1
                mask[:,-1] = 0
        else:
            assert False, "Unknown decode type"
        assert (env.initial_inventory>=0).all(),'can\'t show products with no remaining inventory'
        return assortment[:,:-1].cpu(), entropy ,ass_log_softmax

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type
    
    def save_model(self, args):
        torch.save(
            self.state_dict(),
            args.stream+'BestNet.pt'
        )
        self.args.logger.info('model weights saved')

    def load_weights(self ,args):
        self.load_state_dict(
            torch.load(args.stream+'BestNet.pt')
        )
        print('model weights loaded')

    def test_env(self,env,input_sequence):
        R = torch.zeros([env.batch_size, 1])
        change_of_R = []
        change_of_inv = []
        i = 0
        test_value = []
        if self.args.detail:
            offer_matrix = np.zeros((self.args.num_products+1,110))
            purchase_matrix = np.zeros((self.args.num_products+1,110))
        while not (env.all_finished()):
            arriving_seg = torch.tensor(input_sequence[:, i]).to(self.device)
            cus_type_embedding = self.cus_embedding(arriving_seg)
            s = torch.tensor(env.inventory_level/env.initial_inventory).to(self.device)
            #s = s*self.t_embedding(torch.tensor(env.arrivals).to(self.device))
            s = torch.cat((s,self.t_embedding(torch.tensor(env.arrivals).to(self.device).unsqueeze(0))), dim=1)
            #s = torch.cat((s,torch.tensor([[env.arrivals/env.total_inv]]).to(self.device)), dim=1)
            s = torch.cat((s,cus_type_embedding), dim=1)
            # 关键语句
            x, value = self.forward(env.purchase_history,s)
            value = value.cpu()
            test_value.append(np.mean(value.detach().numpy()))
            #########
            mask = torch.from_numpy(env.get_mask())
            # Select the indices of the next nodes in the sequences
            if self.MLP:
                assortment, p, ass_log_softmax = self._select_node_MLP(env,
                                x, mask.bool())  # Squeeze out steps dimension
            else:
                assortment, p, ass_log_softmax = self._select_node_RNN(env,
                                x, mask.bool())  # Squeeze out steps dimension
            index_, reward = env.step(arriving_seg, assortment.numpy())
            R += reward
            if self.args.detail:
                change_of_inv.append(env.inventory_level.mean(0))
                change_of_R.append(R.numpy().mean(0)[0])
                offer_matrix[assortment[0].nonzero().ravel(),i] += 1
                purchase_matrix[index_,i] += 1
            i += 1
        if self.args.detail:
            return R,test_value,change_of_inv,change_of_R,offer_matrix,purchase_matrix
        return R,test_value







