from net import A2C
from net_DQN import DQN
from func import *
import torch
import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)
from env import market_dynamic
from other_agents import OA_agent,myopic_agent,E_IB_agent,optmyopic_agent,optEIB_agent,sub_t_agent,DP_Greedy_agent,DP_Rollout_agent
import torch
from numpy import *
from func import compute_returns
import os


def train(best,i,args,simulator_list,mnl_etas_list,products_price,train_sequences,logger):
    initial_inventory = np.array([args.ini_inv] * args.num_products)
    torch.manual_seed(args.net_seed+i)
    random.seed(args.net_seed+i)

    if args.DQN:
        seller = DQN(args, mnl_etas_list,i)
        lr_scheduler,optimizer = initialize_DQN(args,seller)
    else:
        if args.cus_onehot:
            seller = A2C_CusOnehot(args).to(args.device)
        else:
            seller = A2C(args).to(args.device)
        lr_scheduler,optimizer = initialize(args,seller)
    if args.load:
        seller.load_weights(args)
    if not args.DQN:
        seller.set_decode_type('greedy')

    env = market_dynamic(args, simulator_list, products_price)

    total_val_reward = val(args,env,seller,initial_inventory,train_sequences)
    logger.info("initial mean reward: {:.4f}".format(total_val_reward))
    
    for epoch in range(args.epoch_num):
        logger.info("start epoch: {} / {}".format(epoch+1,args.epoch_num))
        if args.DQN:
            logger.info("epsilon: {}".format(seller.epsilon))
            logger.info("learning rate now: {}".format(optimizer.param_groups[0]['lr']))
        else:
            logger.info("learning rate now: {},{},{}".format(optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],
                                                             optimizer.param_groups[2]['lr']))
        seq_indexlist = np.arange(len(train_sequences))
        np.random.shuffle(seq_indexlist)
        for seq_index in seq_indexlist:
            input_sequence = train_sequences[seq_index]
            T = len(input_sequence)
            input_sequence = np.array([input_sequence])
            env.reset(initial_inventory, T)
            if args.DQN:
                total_reward = train_batch_DQN(args, seller, lr_scheduler, optimizer, env, input_sequence)
            else:
                if not args.GR_Train:
                    seller.set_decode_type('sampling')
                total_reward = train_batch(args, seller, lr_scheduler, optimizer, env, input_sequence)
        if not args.DQN:
            seller.set_decode_type('greedy')
        total_val_reward = val(args,env,seller,initial_inventory,train_sequences)
        logger.info("mean validate reward: {:.4f}".format(total_val_reward))
        if total_val_reward>best:
            seller.save_model(args)
            best = total_val_reward
    return best

def train_batch(args,
    model, lr_scheduler, optimizer, env, input_sequence):
    # Evaluate model, get costs and log probabilities
    i = 0
    total_reward = 0
    while not (env.all_finished()):
        log_probs, values, rewards, mean_entropy, m_dones, i, next_value = \
            model.roll_out(env, input_sequence, i)
        total_reward += rewards.sum(1).mean()
        #returns = compute_returns(next_value, rewards, m_dones)#包括了部分真实（实施action之后的）的回报，而values全是虚假
        #returns = torch.cat(returns,1).detach()#batch_size*args.num_steps
        #advantage = returns - values#大于0表示action是好
        #actor_loss = -(log_probs * advantage.detach()).mean()
        
        returns, returns2 = compute_returns(next_value, rewards, m_dones, values)
        
        returns2 = torch.cat(returns2,1).detach()
        advantage2 = returns2 - values
        
        returns = torch.cat(returns,1).detach()#batch_size*args.num_steps
        advantage = returns - values#大于0表示action是好

        #actor_loss = -(log_probs * advantage.detach()).mean()
        actor_loss = -(log_probs * advantage2.detach()).mean()
        
        critic_loss = advantage.pow(2).mean()
        loss = args.a_rate*actor_loss + args.c_rate * critic_loss \
               - args.e_rate * mean_entropy#2,0.5,30
        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if optimizer.param_groups[0]['lr']>args.lr_min:
            lr_scheduler.step()
    return total_reward
    
def train_batch_DQN(args,
    model, lr_scheduler, optimizer, env, input_sequence,train_=True):
    #给一个batch的input_sequence，训练模型
    total_reward = 0
    i = 0
    arriving_seg = input_sequence[:, i]

    input = torch.tensor(env.inventory_level/env.initial_inventory)
    input = torch.cat((input, model.t_embedding(torch.tensor(env.arrivals).unsqueeze(0))), dim=1)
    cus_type_embedding = model.cus_embedding(torch.tensor(arriving_seg))
    input = torch.cat((input,cus_type_embedding), dim=1)

    s = torch.from_numpy(env.inventory_level/env.initial_inventory)#shape(1,...)
    s = torch.cat((s,torch.tensor([[env.arrivals]])), dim=1)
    state = torch.cat((s,torch.from_numpy(arriving_seg).reshape(1,-1)), dim=1)

    
    while not (env.all_finished()):
        mask = env.get_mask()
        ###############################
        assortment = model.select_action(mask,input,arriving_seg,train_)###########
        ###############################
        _, reward = env.step(arriving_seg, assortment)
        total_reward += reward[0]
        i +=1
        next_arriving_seg = input_sequence[:, i]

        input = torch.tensor(env.inventory_level/env.initial_inventory)
        input = torch.cat((input, model.t_embedding(torch.tensor(env.arrivals).unsqueeze(0))), dim=1)
        cus_type_embedding = model.cus_embedding(torch.tensor(next_arriving_seg))
        input = torch.cat((input,cus_type_embedding), dim=1)
        
        s = torch.from_numpy(env.inventory_level/env.initial_inventory)#shape(1,...)
        s = torch.cat((s,torch.tensor([[env.arrivals]])), dim=1)
        next_state = torch.cat((s,torch.from_numpy(next_arriving_seg).reshape(1,-1)), dim=1)
            
        if bool(train_):
            model.remember(state[0], assortment[0], reward[0], next_state[0], env.all_finished())#state is tensor
            if model.replaySize > 1000:
                model.update(optimizer,lr_scheduler)
        state = next_state
        arriving_seg = next_arriving_seg
    return total_reward
    
def val(args,env,seller,initial_inventory,train_sequences):
    env.batch_size = args.batch_size
    env.initial_inventory = np.tile(initial_inventory, (args.batch_size, 1))
    env.inventory_level = np.tile(initial_inventory, (args.batch_size, 1))
    
    seller_list = np.zeros((env.batch_size, 1))
    
    seq_indexlist = np.arange(len(train_sequences))
    for seq_index in seq_indexlist:
        input_sequence = train_sequences[seq_index]
        T = len(input_sequence)
        input_sequence = np.array([input_sequence])
        env.reset(initial_inventory, T)
        #####################
        if args.DQN:
            cost = train_batch_DQN(args, seller, 0, 0, env, input_sequence,train_=False)
        else:
            cost,test_value = seller.test_env(env, input_sequence)
        #####################
        seller_list = np.vstack((seller_list, cost))
    seller_list = list(seller_list.ravel()[env.batch_size:])
    env.batch_size = args.batch_size
    env.initial_inventory = np.tile(initial_inventory, (args.batch_size, 1))
    env.inventory_level = np.tile(initial_inventory, (args.batch_size, 1))
    return mean(seller_list)


def test(test_sequences,GT_choice_model_list,mnl_list,products_price,args,logger,only_test_benchmark=False,only_test_DRL=False):
    initial_inventory = np.array([args.ini_inv] * args.num_products)
    for seed_ in range(args.seed_range):
        
        env_OA = market_dynamic(args,GT_choice_model_list,products_price)
        env_myopic = market_dynamic(args,GT_choice_model_list,products_price)
        env_EIB = market_dynamic(args,GT_choice_model_list,products_price)
        env_sub = market_dynamic(args,GT_choice_model_list,products_price)
        env_DP_Greedy = market_dynamic(args,GT_choice_model_list,products_price)
        env_DP_Rollout = market_dynamic(args,GT_choice_model_list,products_price)
        
        OA_seller = OA_agent(args,env_OA, products_price)
        myopic_seller = myopic_agent(args,env_myopic,mnl_list,products_price)
        E_IB_seller = E_IB_agent(args,env_EIB,mnl_list,products_price)
        sub_t_seller = sub_t_agent(args,env_sub,mnl_list,products_price)
        DP_Greedy_seller = DP_Greedy_agent(args,env_DP_Greedy,mnl_list,products_price)
        DP_Rollout_seller = DP_Rollout_agent(args,env_DP_Rollout,mnl_list,products_price)
        
        OA_list = np.zeros((args.batch_size, 1))
        myopic_list = np.zeros((args.batch_size, 1))
        E_IB_list = np.zeros((args.batch_size, 1))
        sub_t_list = np.zeros((args.batch_size, 1))
        DP_Greedy_list = np.zeros((args.batch_size, 1))
        DP_Rollout_list = np.zeros((args.batch_size, 1))

        env = market_dynamic(args,GT_choice_model_list,products_price)
        if args.cus_onehot:
            seller = A2C_CusOnehot(args).to(args.device)
        else:
            seller = A2C(args).to(args.device)
        if (not only_test_benchmark) and load:
            seller.load_weights(args)
        if args.GR_Test:
            seller.set_decode_type('greedy')
        else:
            seller.set_decode_type('sampling')
        seller_list = np.zeros((args.batch_size, 1))
        test_value_list=[]
        change_of_inv_list=[]
        change_of_R_list=[]

        seq_indexlist = np.arange(len(test_sequences))
        if args.detail:
            offer_matrix_sum,purchase_matrix_sum = np.zeros((args.num_products+1,110)),np.zeros((args.num_products+1,110))
            
        for seq_index in seq_indexlist:
            input_sequence = test_sequences[seq_index]
            T = len(input_sequence)
            input_sequence = np.array([input_sequence])
            if only_test_benchmark:
                other_agents(OA_seller, myopic_seller, E_IB_seller,sub_t_seller,DP_Greedy_seller,DP_Rollout_seller,initial_inventory, T, products_price, input_sequence)
                cost = np.zeros((args.batch_size, 1))
            else:
                env.reset(initial_inventory, T)
                if args.detail:
                    cost,test_value,change_of_inv,change_of_R,offer_matrix,purchase_matrix = seller.test_env(env, input_sequence)
                    test_value_list.append(np.pad(np.array(test_value) ,(0,110-T),'edge' ))
                    change_of_inv_list.append(np.pad(np.array(change_of_inv) ,((0,110-T),(0,0)),'edge' ))
                    change_of_R_list.append(np.pad(np.array(change_of_R) ,(0,110-T),'edge' ))
                    offer_matrix_sum += offer_matrix
                    purchase_matrix_sum += purchase_matrix
                else:
                    cost,test_value = seller.test_env(env, input_sequence)
            OA_list = np.vstack((OA_list,OA_seller.total_reward))
            myopic_list = np.vstack((myopic_list, myopic_seller.total_reward))
            E_IB_list = np.vstack((E_IB_list, E_IB_seller.total_reward))
            sub_t_list = np.vstack((sub_t_list, sub_t_seller.total_reward))
            DP_Greedy_list = np.vstack((DP_Greedy_list, DP_Greedy_seller.total_reward))
            DP_Rollout_list = np.vstack((DP_Rollout_list, DP_Rollout_seller.total_reward))
            seller_list = np.vstack((seller_list, cost))
        if args.detail:
            folder = 'log/' + args.folder+'/DRLTrainLog'+ args.saved_time_stream
            os.makedirs(folder + 'change', exist_ok=True)
            np.save(folder + 'change/test_value_list'+str(seed_)+'.npy',np.array(test_value_list))
            np.save(folder + 'change/change_of_inv_list'+str(seed_)+'.npy',np.array(change_of_inv_list))
            np.save(folder + 'change/change_of_R_list'+str(seed_)+'.npy',np.array(change_of_R_list))
            offer_matrix_sum = offer_matrix_sum/len(seq_indexlist)
            purchase_matrix_sum = purchase_matrix_sum/len(seq_indexlist)
            np.save(folder + 'change/offer_matrix'+str(seed_)+'.npy',np.array(offer_matrix))
            np.save(folder + 'change/purchase_matrix'+str(seed_)+'.npy',np.array(purchase_matrix))
        OA_list = list(OA_list.ravel()[args.batch_size:])
        myopic_list = list(myopic_list.ravel()[args.batch_size:])
        E_IB_list = list(E_IB_list.ravel()[args.batch_size:])
        sub_t_list = list(sub_t_list.ravel()[args.batch_size:])
        DP_Greedy_list = list(DP_Greedy_list.ravel()[args.batch_size:])
        DP_Rollout_list = list(DP_Rollout_list.ravel()[args.batch_size:])
        seller_list = list(seller_list.ravel()[args.batch_size:])
        logger.info("mean test reward1: {}".format(seller_list))
        logger.info("mean test reward2: {}".format(OA_list))
        logger.info("mean test reward3: {}".format(myopic_list))
        logger.info("mean test reward4: {}".format(E_IB_list))
        logger.info("mean test reward5: {}".format(sub_t_list))
        logger.info("mean test reward6: {}".format(DP_Greedy_list))
        logger.info("mean test reward7: {}".format(DP_Rollout_list))
        logger.info("mean test reward: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(mean(seller_list), mean(OA_list),
              mean(myopic_list), mean(E_IB_list), mean(sub_t_list), mean(DP_Greedy_list), mean(DP_Rollout_list)))


def other_agents(OA_seller,myopic_seller,E_IB_seller,sub_t_seller,DP_Greedy_seller,DP_Rollout_seller,
                 initial_inventory,T,products_price,input_sequence):
    OA_seller.reset(initial_inventory, T)
    myopic_seller.reset(initial_inventory, T, products_price)
    E_IB_seller.reset(initial_inventory, T)
    sub_t_seller.reset(initial_inventory, T, products_price)
    DP_Greedy_seller.reset(initial_inventory, T, products_price)
    DP_Rollout_seller.reset(initial_inventory, T, products_price)
    for t in range(T-1):
        arriving_seg = input_sequence[:, t].reshape(-1, 1)
        OA_seller.step(arriving_seg)
        myopic_seller.step(arriving_seg)
        E_IB_seller.step(arriving_seg)
        sub_t_seller.step(arriving_seg)
        DP_Greedy_seller.step(arriving_seg)
        DP_Rollout_seller.step(arriving_seg)

def test_DQN(test_sequences,GT_choice_model_list,mnl_etas_list,initial_inventory,products_price,args,logger,load):
    for seed_ in range(args.seed_range):
        env = market_dynamic(args,GT_choice_model_list,products_price)
        seller = DQN(args, mnl_etas_list,seed_)
        if load:
            seller.load_weights(args)
        seller_list = np.zeros((args.batch_size, 1))
        seq_indexlist = np.arange(len(test_sequences))
        check = False
        for seq_index in seq_indexlist:
            input_sequence = test_sequences[seq_index]
            T = len(input_sequence)
            input_sequence = np.array([input_sequence])
            env.reset(initial_inventory, T)
            cost = train_batch_DQN(args,seller,0,0,env,input_sequence,train_=False)
            seller_list = np.vstack((seller_list, cost))
        seller_list = list(seller_list.ravel()[args.batch_size:])
        logger.info("mean test reward1: {}".format(seller_list))
        logger.info("mean test reward: {:.4f}".format(mean(seller_list)))

def initialize(args,model):
    if not args.cus_onehot:
        optimizer = optim.Adam(
            [{"params": model.share.parameters(),"lr": args.share_lr},
             {"params": model.actor.parameters(), "lr": args.actor_lr},
             {"params": model.critic.parameters(), "lr": args.critic_lr},
             {'params': model.cus_embedding.parameters(), 'lr': args.em_lr},
             {'params': model.t_embedding.parameters(), 'lr': args.em_lr},]
        )
    else:
        optimizer = optim.Adam(
            [{"params": model.share.parameters(),"lr": args.share_lr},
             {"params": model.actor.parameters(), "lr": args.actor_lr},
             {"params": model.critic.parameters(), "lr": args.critic_lr},
             {'params': model.t_embedding.parameters(), 'lr': args.em_lr},]
        )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.step,gamma=args.lr_decay_lambda
    )
    return lr_scheduler,optimizer


def initialize_DQN(args,model):
    dict_ = {}
    dict_["params"] = model.QNetwork.parameters()
    dict_["lr"] = args.lr
    optimizer = optim.Adam([dict_,
                           {'params': model.cus_embedding.parameters(), 'lr': args.em_lr},
                           {'params': model.t_embedding.parameters(), 'lr': args.em_lr}])#model.actor.state_dict()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.step,gamma=args.lr_decay_lambda
    )
    return lr_scheduler,optimizer
    
    
'''

def initialize_DQN(args,model):
    dict_ = {}
    dict_["params"] = model.QNetwork.parameters()
    optimizer = optim.Adam([dict_],lr=args.lr)#model.actor.state_dict()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.step,gamma=args.lr_decay_lambda
    )
    return lr_scheduler,optimizer'''