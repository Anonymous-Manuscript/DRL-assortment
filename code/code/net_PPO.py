import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
import time
from func import compute_returns


class PPO(nn.Module):
    def __init__(self, args):
        super(PPO, self).__init__()

        self.MLP = args.MLP
        self.decode_type = None
        self.num_products = args.num_products
        self.num_cus_types = args.num_cus_types
        self.cardinality = args.cardinality
        self.args = args
        self.device = args.device

        # customer type
        self.cus_embedding_dim = args.cus_embedding_dim
        self.cus_embedding = nn.Embedding(args.num_cus_types, self.cus_embedding_dim)
        # time
        self.t_embedding_dim = args.t_embedding_dim
        self.t_embedding = nn.Embedding(args.ini_inv * args.num_products * 5, self.t_embedding_dim)

        share = []
        share.append(nn.Linear(args.num_products + self.cus_embedding_dim + self.t_embedding_dim, args.w[0]))  #
        share.append(nn.ReLU())
        for h in range(args.h - 1):
            share.append(nn.Linear(args.w[h], args.w[h + 1]))
            share.append(nn.ReLU())
        share.append(nn.Linear(args.w[args.h - 1], args.nn_out))
        share.append(nn.ReLU())
        self.share = nn.Sequential(*share)

        self.critic = nn.Sequential(nn.Linear(args.nn_out, 1))

        if self.MLP:
            self.actor = nn.Sequential(
                *[nn.Linear(args.nn_out, args.num_products)]
                # ,nn.ReLU(),nn.Linear(args.num_products, args.num_products)]
                # ,nn.ReLU(),nn.Linear(args.num_products, args.num_products)]
            )
        else:
            self.actor = nn.RNN(args.nn_out, args.num_products + 1, 1)  ###########

        self.clip_rate = 0.5
        self.e_rate = args.e_rate
        self.train_repeats = 5

    def forward(self, x):
        x = self.share(x)
        value = self.critic(x)
        if not self.MLP:
            return x, value
        else:
            score = self.actor(x)  ##########
            return score, value  ##########

    def train_batch(self, args, lr_scheduler, optimizer, env, input_sequence):
        self.decode_type = "sampling"
        i = 0
        total_reward = 0
        while not (env.all_finished()):
            states0, actions, masks, old_log_probs, values, rewards, mean_entropy, m_dones, i, next_value = self.roll_out(env, input_sequence, i)
            
            _, returns = compute_returns(next_value, rewards, m_dones, values)

            returns = torch.cat(returns, 1).detach()
            advantage = returns - values.detach()
            old_log_probs = old_log_probs.detach()

            for num_step in range(self.train_repeats):
                states = self.Embedding_state(states0)
                new_entropy, ratio, new_values = self.rollout_again(states, actions, masks, old_log_probs)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * advantage

                actor_loss = -torch.min(surr1, surr2) - self.e_rate * new_entropy
                critic_loss = (returns - new_values).pow(2).mean()
                loss = args.a_rate * actor_loss + args.c_rate * critic_loss
                loss = loss.mean()

                # Perform backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if optimizer.param_groups[0]['lr'] > args.lr_min:
                    lr_scheduler.step()

    def Embedding_state(self,states):
        cat_states = torch.cat(states, 0)
        Embedding_state_return = cat_states[:,:-2]
        
        Embedding_state_return = torch.cat((Embedding_state_return, self.t_embedding(cat_states[:,-2].int()) ), dim=1)
        Embedding_state_return = torch.cat((Embedding_state_return, self.cus_embedding(cat_states[:,-1].int()) ), dim=1)

        return Embedding_state_return

    def roll_out(self, env, input_sequence, i):
        ass_log_softmax_list_ = []
        values = []
        R = []
        m_dones = []
        states = []
        actions = []
        masks = []
        mean_entropy = 0
        s = 0
        for num_step in range(self.args.num_steps):
            arriving_seg = torch.tensor(input_sequence[:, i]).to(self.device)
            cus_type_embedding = self.cus_embedding(arriving_seg)
            s = torch.tensor(env.inventory_level / env.initial_inventory).to(self.device)
            state = s
            # s = s*self.t_embedding(torch.tensor(env.arrivals).to(self.device))
            s = torch.cat((s, self.t_embedding(torch.tensor(env.arrivals).to(self.device).unsqueeze(0))), dim=1)
            # s = torch.cat((s,torch.tensor([[env.arrivals/env.total_inv]]).to(self.device)), dim=1)
            s = torch.cat((s, cus_type_embedding), dim=1)
            
            state = torch.cat((state, torch.tensor(env.arrivals).to(self.device).reshape((1,1)) ), dim=1)
            state = torch.cat((state, arriving_seg.reshape((1,1)) ), dim=1)
            states.append(state)
            # 关键语句
            x, value = self.forward(s)
            #########
            mask = torch.from_numpy(env.get_mask())
            masks.append(mask)
            # Select the indices of the next nodes in the sequences
            action, assortment, entropy, ass_log_softmax = self._select_node_RNN(env, x, mask.bool())  # Squeeze out steps dimension
            actions.append(action)

            mean_entropy += entropy
            _, reward = env.step(arriving_seg, assortment.numpy())
            # 要输出的东西
            ass_log_softmax_list_.append(ass_log_softmax)
            values.append(value)
            m_dones.append(1 - env.all_finished())
            R.append(torch.DoubleTensor([reward]))
            i += 1
            if num_step == self.args.num_steps - 1 or env.all_finished():  # 这个roll out结束
                if env.all_finished():
                    next_value = 0  # done之后这个值是没用的
                    break
                arriving_seg = torch.randint(1, self.num_cus_types, (1,)).to(self.device)
                cus_type_embedding = self.cus_embedding(arriving_seg)
                next_state = torch.tensor(env.inventory_level / env.initial_inventory).to(self.device)
                # next_state = next_state*self.t_embedding(torch.tensor(env.arrivals).to(self.device))
                next_state = torch.cat(
                    (next_state, self.t_embedding(torch.tensor(env.arrivals).to(self.device).unsqueeze(0))), dim=1)
                # next_state = torch.cat((next_state,torch.tensor([[env.arrivals/env.total_inv]]).to(self.device)), dim=1)
                next_state = torch.cat((next_state, cus_type_embedding), dim=1)
                _, next_value = self.forward(next_state)
                break
        # Collected lists, return Tensor
        return (
            states,
            actions,
            masks,
            torch.stack(ass_log_softmax_list_, 1),  # shape(batch_size,T)
            torch.cat(values, 1),
            torch.cat(R, 1).to(self.device),  ###
            mean_entropy.double(),
            torch.tensor(m_dones).to(self.device),
            i,
            next_value
        )

    def rollout_again(self, states, actions, masks, old_log_probs):
        new_entropys = []
        ratios = []
        x, new_values = self.forward(states)

        for i in range(len(actions)):
            scores, hidden_layer = self.actor(x[i].repeat(self.args.cardinality, 1).reshape(self.args.cardinality, 1, -1))
            entropy = 0
            new_log_prob = 0
            action = actions[i]
            mask = masks[i]
            mask = torch.cat((mask, torch.tensor([[1]])), -1).to(self.device)
            for node_num,selected in enumerate(action):
                score = scores[node_num]
                score = score.masked_fill(mask.bool(), value=torch.tensor(-1e20))
                p = torch.log_softmax(score, dim=1)
                dist = Categorical(p)
                entropy += dist.entropy().mean()
                
                new_log_prob += p[:,selected[0][0]][0]
                
                mask[:, selected[0][0]] = 1
                mask[:, -1] = 0
            ratio = torch.exp(new_log_prob - old_log_probs[:,i][0])
            new_entropys.append(entropy)
            ratios.append(ratio)

        return torch.stack(new_entropys).reshape((1,-1)), torch.stack(ratios).reshape((1,-1)), new_values.reshape((1,-1))

    def calc_entropy(self, _log_p):
        entropy = -(_log_p * _log_p.exp()).sum(2).sum(1).mean()
        return entropy

    def _select_node_MLP(self, env, score, mask):
        score[mask] = -1e20
        p = torch.log_softmax(score, dim=1)
        dist = Categorical(p)
        entropy = dist.entropy().mean()  # 想让 entropy 变大，更加随机
        ass = torch.zeros([env.batch_size, self.num_products], dtype=torch.int)
        if self.decode_type == "greedy":
            _, idx1 = torch.sort(p, descending=True)  # descending为False，升序，为True，降序
            selected = idx1[:, :self.cardinality]
            ass.scatter_(1, selected, 1)
            ass = ass * torch.logical_not(mask)  # 注意：不是mask着的东西才能摆
            ass_log_softmax = (ass * p).sum(axis=1)  # 是一个长度为batch_size的张量
        elif self.decode_type == "sampling":
            selected = p.exp().multinomial(self.cardinality, replacement=True)  # 放回的抽样
            ass.scatter_(1, selected, 1)
            ass = ass * torch.logical_not(mask)  # 注意：不是mask着的东西才能摆
            ass_log_softmax = (ass * p).sum(axis=1)
        else:
            assert False, "Unknown decode type"
        return ass, entropy, ass_log_softmax

    def _select_node_RNN(self, env, x, mask):
        action = []
        assortment = torch.zeros([env.batch_size, self.num_products + 1], dtype=torch.int).to(self.device)
        entropy = 0
        ass_log_softmax = 0
        mask = torch.cat((mask, torch.tensor([[1]])), -1).to(self.device)
        if self.decode_type == "greedy":
            scores, hidden_layer = self.actor(x.repeat(self.args.cardinality, 1).reshape(self.args.cardinality, 1, -1))
            for node_num in range(self.args.cardinality):
                score = scores[node_num]
                score = score.masked_fill(mask.bool(), value=torch.tensor(-1e20))
                p = torch.log_softmax(score, dim=1)
                dist = Categorical(p)
                entropy += dist.entropy().mean()  # 想让 entropy 变大，更加随机

                _, idx1 = torch.sort(p, descending=True)  # descending为False，升序，为True，降序
                selected = idx1[:, :1]

                ass = torch.zeros([env.batch_size, self.num_products + 1], dtype=torch.int).to(self.device).scatter(1,
                                                                                                                    selected,
                                                                                                                    1)
                ass_log_softmax += (ass * p).sum(axis=1)  # 是一个长度为batch_size的张量
                if selected == self.num_products:
                    break
                assortment += ass
                mask[:, selected[0][0]] = 1
                mask[:, -1] = 0
            # ass = ass*torch.logical_not(mask)#注意：不是mask着的东西才能摆

        elif self.decode_type == "sampling":
            scores, hidden_layer = self.actor(x.repeat(self.args.cardinality, 1).reshape(self.args.cardinality, 1, -1))
            for node_num in range(self.args.cardinality):
                score = scores[node_num]
                score = score.masked_fill(mask.bool(), value=torch.tensor(-1e20))
                p = torch.log_softmax(score, dim=1)
                dist = Categorical(p)
                entropy += dist.entropy().mean()  # 想让 entropy 变大，更加随机
                selected = p.exp().multinomial(1)
                action.append(selected)

                ass = torch.zeros([env.batch_size, self.num_products + 1], dtype=torch.int).to(self.device).scatter(1,
                                                                                                                    selected,
                                                                                                                    1)
                ass_log_softmax += (ass * p).sum(axis=1)  # 是一个长度为batch_size的张量
                if selected == self.num_products:
                    break
                assortment += ass
                mask[:, selected[0][0]] = 1
                mask[:, -1] = 0
        else:
            assert False, "Unknown decode type"
        assert (env.initial_inventory >= 0).all(), 'can\'t show products with no remaining inventory'
        return action, assortment[:, :-1].cpu(), entropy, ass_log_softmax

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def save_model(self, args):
        torch.save(
            self.state_dict(),
            'PPO' + args.stream + 'BestNet.pt'
        )
        self.args.logger.info('model weights saved')

    def load_weights(self, args):
        self.load_state_dict(
            torch.load('PPO' + args.stream + 'BestNet.pt')
        )
        print('model weights loaded')

    def test_env(self, env, input_sequence):
        R = torch.zeros([env.batch_size, 1])
        change_of_R = []
        change_of_inv = []
        i = 0
        test_value = []
        if self.args.detail:
            offer_matrix = np.zeros((self.args.num_products + 1, 110))
            purchase_matrix = np.zeros((self.args.num_products + 1, 110))
        while not (env.all_finished()):
            arriving_seg = torch.tensor(input_sequence[:, i]).to(self.device)
            cus_type_embedding = self.cus_embedding(arriving_seg)
            s = torch.tensor(env.inventory_level / env.initial_inventory).to(self.device)
            # s = s*self.t_embedding(torch.tensor(env.arrivals).to(self.device))
            s = torch.cat((s, self.t_embedding(torch.tensor(env.arrivals).to(self.device).unsqueeze(0))), dim=1)
            # s = torch.cat((s,torch.tensor([[env.arrivals/env.total_inv]]).to(self.device)), dim=1)
            s = torch.cat((s, cus_type_embedding), dim=1)
            # 关键语句
            x, value = self.forward(s)
            value = value.cpu()
            test_value.append(np.mean(value.detach().numpy()))
            #########
            mask = torch.from_numpy(env.get_mask())
            # Select the indices of the next nodes in the sequences
            if self.MLP:
                assortment, p, ass_log_softmax = self._select_node_MLP(env,
                                                                       x, mask.bool())  # Squeeze out steps dimension
            else:
                _, assortment, p, ass_log_softmax = self._select_node_RNN(env,
                                                                       x, mask.bool())  # Squeeze out steps dimension
            index_, reward = env.step(arriving_seg, assortment.numpy())
            R += reward
            if self.args.detail:
                change_of_inv.append(env.inventory_level.mean(0))
                change_of_R.append(R.numpy().mean(0)[0])
                offer_matrix[assortment[0].nonzero().ravel(), i] += 1
                purchase_matrix[index_, i] += 1
                # breakpoint()
            i += 1
        if self.args.detail:
            return R, test_value, change_of_inv, change_of_R, offer_matrix, purchase_matrix
        return R, test_value

