import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
import time

def E_penalty_function(x):
    return (1-np.exp(-x))*(np.e/(np.e-1))
    
class PG(nn.Module):
    def __init__(self, args):
        super(PG, self).__init__()
        
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
        
        args.lr_decay_lambda = 0.995
        
        share = []
        share.append(nn.Linear(args.num_products+self.cus_embedding_dim+self.t_embedding_dim, args.w[0]))#
        share.append(nn.ReLU())
        for h in range(args.h-1):
            share.append(nn.Linear(args.w[h], args.w[h+1]))
            share.append(nn.ReLU())
        share.append(nn.Linear(args.w[args.h-1], args.nn_out))
        share.append(nn.ReLU())
        self.share = nn.Sequential(*share)

        if self.MLP:
            self.actor = nn.Sequential(
                 *[nn.Linear(args.nn_out, args.num_products)]#,nn.ReLU(),nn.Linear(args.num_products, args.num_products)]
                   #,nn.ReLU(),nn.Linear(args.num_products, args.num_products)]
            )
        else:
            self.actor = nn.RNN(args.nn_out, args.num_products+1,1)   ###########
        
    def forward(self, x):
        x = self.share( x )
        if not self.MLP:
            return x
        else:
            score = self.actor(x)##########
            return score

    def train_batch(self, args, lr_scheduler, optimizer, env, input_sequence):
        self.decode_type = "sampling"
        
        total_reward, log_prob_sum, mean_entropy = self.roll_out(env, input_sequence)
        
        actor_loss = -torch.from_numpy(total_reward) * log_prob_sum
        
        loss = args.a_rate * actor_loss - args.e_rate * mean_entropy
        loss = loss.mean()

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if optimizer.param_groups[0]['lr'] > args.lr_min:
            lr_scheduler.step()

    
    def roll_out(self, env, input_sequence):
        total_reward = 0
        log_prob_sum = 0
        mean_entropy = 0
        s = 0
        i = 0
        while not (env.all_finished()):
            arriving_seg = torch.tensor(input_sequence[:, i]).to(self.device)
            cus_type_embedding = self.cus_embedding(arriving_seg)
            s = torch.tensor(env.inventory_level/env.initial_inventory).to(self.device)
            #s = s*self.t_embedding(torch.tensor(env.arrivals).to(self.device))
            s = torch.cat((s,self.t_embedding(torch.tensor(env.arrivals).to(self.device).unsqueeze(0))), dim=1)
            #s = torch.cat((s,torch.tensor([[env.arrivals/env.total_inv]]).to(self.device)), dim=1)
            s = torch.cat((s,cus_type_embedding), dim=1)
            # 关键语句
            x = self.forward(s)
            #########
            mask = torch.from_numpy(env.get_mask())
            # Select the indices of the next nodes in the sequences
            if self.MLP:
                assortment, entropy, ass_log_softmax = self._select_node_MLP(env,
                                x, mask.bool())  # Squeeze out steps dimension
            else:
                assortment, entropy, ass_log_softmax = self._select_node_RNN(env,
                                x, mask.bool())  # Squeeze out steps dimension
            _, reward = env.step(arriving_seg,assortment.numpy())
            #要输出的东西
            mean_entropy += entropy
            total_reward += reward
            log_prob_sum += ass_log_softmax
            i += 1
        # Collected lists, return Tensor
        return (
            total_reward,#shape(batch_size,T)
            log_prob_sum.double(),###
            mean_entropy.double()
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
            'PG' + args.stream+'BestNet.pt'
        )
        self.args.logger.info('model weights saved')

    def load_weights(self ,args):
        self.load_state_dict(
            torch.load('PG' + args.stream+'BestNet.pt')
        )
        print('model weights loaded')

    def test_env(self,env,input_sequence):
        R = torch.zeros([env.batch_size, 1])
        change_of_R = []
        change_of_inv = []
        i = 0
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
            x = self.forward(s)
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
                #breakpoint()
            i += 1
        if self.args.detail:
            return R,change_of_inv,change_of_R,offer_matrix,purchase_matrix
        return R,1



