from func import setup_logger
import torch
import numpy as np
from arg import init_parser
import os
import time
import json
from models.ranked_list import RankedListModel
from models.GSP import GSPModel
from models.multinomial import MultinomiallogitModel
from models.Assort_net import Gate_Assort_Net
from models.latent_class import LatentClassModel
from models.mmnl import MixedMNLModel
from models.rcs import RcsModel
from models.exponomial import ExponomialModel
from models.mc import MarkovChainModel
from GT.transactions_arrival import Transaction
########################################################
###  read args and set logger
########################################################
parser = init_parser('A2C')
args = parser.parse_args()
device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
args.device = device
args.device = torch.device("cpu")

# Assuming args.folder contains the folder name
network = 'RNN'
if args.MLP:
    network = 'MLP'
elif args.PPO:
    network = 'PPO'
elif args.PG:
    network = 'PG'
    args.epoch_num = 100
GT_folder = args.GT+ 'N=' + str(args.num_products) + 'M=' + str(args.num_cus_types) + 'k=' + str(args.k) + 'L=' + str(args.L) + 'T=' + str(args.T)
if args.DQN:
    args.folder = args.GT+'N='+str(args.num_products)+'M='+str(args.num_cus_types)+\
              'k='+str(args.k)+'L='+str(args.L)+'T='+str(args.T)+'INV='+str(args.ini_inv)+'DQN'+args.CM
else:
    args.folder = args.GT+'N='+str(args.num_products)+'M='+str(args.num_cus_types)+\
              'k='+str(args.k)+'L='+str(args.L)+'T='+str(args.T)+'INV='+str(args.ini_inv)+network+args.CM
# Create the directory if it doesn't exist
os.makedirs('log/'+args.folder, exist_ok=True)
#os.makedirs('DRL_save/'+args.folder, exist_ok=True)

if (not args.UB) and (not args.only_test_Benchmark):
    time_stream = time.strftime('%Y-%m-%d-%H-%M-%S')
    if args.DQN:
        args.stream = 'DQN_save/'+args.folder+time_stream # for saving the trained model
    else:
        args.stream = 'DRL_save/'+args.folder+time_stream # for saving the trained model
    stream = os.path.join('log', args.folder+'/DRLTrainLog'+ time_stream)
    logger = setup_logger(name='Train', level=20, stream = stream)
    args.logger = logger
    logger.info(args)
    logger.info(stream)

########################################################
###  set experiment parameters
########################################################
args.products = [i for i in range(args.num_products+1)]
initial_inventory = np.array([args.ini_inv]*args.num_products)

########################################################
### groundtruth model
########################################################
products_price = np.load('GT/'+GT_folder+'/prices.npy')
products_price = products_price/100

GT_choice_model_list = [] # multi-type ground truth
# initiate ground truth choice model from the loaded parameters
for i in range(args.num_cus_types):
    if args.GT == 'rl':
        args.rank_list = np.load('GT/' + GT_folder + '/GT_ranked_lists.npy')
        args.cus_type = np.load('GT/' + GT_folder + '/GT_cus_types.npy')
        GT_choice_model = RankedListModel
        data = {'products': args.products}
        data['betas'] = list(args.cus_type[i])
        data['ranked_lists'] = args.rank_list.tolist()
        GT_choice_model_list.append(GT_choice_model.from_data(data))
    elif args.GT == 'lcmnl':
        GT_choice_model = LatentClassModel
        features = GT_choice_model.feature()
        data = {'products': args.products}
        for feature in features:
            file_name = 'GT/' + GT_folder + '/GT_' + feature + '.npy'
            '''    
            data[feature] = np.load(file_name).tolist()[i]
            '''
            if feature=='gammas':
                data[feature] = np.load(file_name).tolist()[i]
            else:
                data[feature] = np.load(file_name).tolist()
            #print(data[feature])
            #breakpoint()
        GT_choice_model_list.append(GT_choice_model.from_data(data))
    elif args.GT == 'gsp':
        GT_choice_model = GSPModel
        args.rank_list = np.load('GT/' + GT_folder + '/GT_ranked_lists.npy')
        args.cus_type = np.load('GT/' + GT_folder + '/GT_cus_types.npy')
        args.k_probs = np.load('GT/' + GT_folder + '/GT_k_probs.npy')
        args.k_list = np.load('GT/' + GT_folder + '/GT_k_list.npy')
        data = {'products': args.products}
        data['betas'] = list(args.cus_type[i])
        data['ranked_lists'] = args.rank_list.tolist()
        data['k_probs'] = list(args.k_probs)
        data['k_list'] = list(args.k_list)
        GT_choice_model_list.append(GT_choice_model.from_data(data))
with open('GT/'+GT_folder+'/seqdata.json', 'r') as f:
    args.seqdata = json.loads(f.read())
with open('GT/'+GT_folder+'/transdata.json', 'r') as f:
    args.transdata = json.loads(f.read())

########################################################
### split train and test
########################################################
train_sequences = list(args.seqdata.values())[:int(0.8*args.T)]
seg_prob = np.zeros(args.num_cus_types+1)
for l in train_sequences:
    seg_prob[-1] += len(l)
    for m in range(args.num_cus_types):
        seg_prob[m] += l.count(m)
args.seg_prob = (seg_prob/seg_prob[-1])[:-1]
test_sequences = list(args.seqdata.values())[int(0.8*args.T):]

if args.UB:
    stream = os.path.join('GT', GT_folder+'/UB_INV'+str(args.ini_inv)+'C'+str(args.cardinality))
    logger = setup_logger(name='Train', level=20, stream = stream)
    args.logger = logger
    #logger.info(args)
    logger.info(stream)
    logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
    from GT.directUB import solve_lp
    #UB_cus_type = np.insert(np.array(args.cus_type), 0, 0.1, axis=1)
    
    solve_current, assort_list = solve_lp(args.num_products,args.cardinality,GT_choice_model_list,products_price,initial_inventory)
    
    #solve_current = solve_lp(args.num_products,args.cardinality,GT_choice_model_list,products_price,initial_inventory)
    
    for seq in train_sequences:
        t1 = time.time()
        UB,y_S_T = solve_current(seq)
        t2 = time.time()
        logger.info('training upper bound:{}'.format(UB))
        logger.info('time for this instance:{}'.format(t2-t1))
        
    for seq in test_sequences:    
        UB, y_S_T = solve_current(seq)
        logger.info('testing upper bound:{}'.format(UB))
    logger.info('upper bound completed.')
    
elif args.UB_assortment:
    from GT.directUB import solve_lp
    solve_current, assort_list = solve_lp(args.num_products,args.cardinality,GT_choice_model_list,products_price,initial_inventory)
    
    assort_list = np.stack(assort_list)
    num_assorts = assort_list.shape[0]
    result = {}
    for every_20 in [0,20,40,60,80]:
        result[every_20] = []
    for seq in test_sequences:
        UB,y_S_T = solve_current(seq)
        for every_20 in [0,20,40,60,80]:
            ass_list = []
            for assort in range(num_assorts):
                ass_list.append(y_S_T[(assort+num_assorts*every_20):(num_assorts*(every_20+20)):num_assorts].sum())
                #for an assortment, the total prob of showing it from time every_20 to every_20+20
            result[every_20].append(ass_list)
    for every_20 in [0,20,40,60,80]:
        argmax_assort = np.argsort(np.array(result[every_20]).sum(axis=0))[::-1][:3]#sum(axis=0) over sequences
        print(assort_list[argmax_assort])
    breakpoint()
    
else:
    ########################################################
    ### load simulator for each type
    ########################################################
    simulator_model_dict = {'lcmnl':LatentClassModel, 'exp':ExponomialModel, 'mc':MarkovChainModel, 'rcs':RcsModel}
    mnl_list = []
    mnl_etas_list = []
    simulator_list = []
    for i in range(args.num_cus_types):
        mnl_etas = np.load('GT/' + GT_folder + '/Cus_type' + str(i) + '/mnl_etas.npy')
        mnl_etas_list.append(mnl_etas)
        mnl_list.append(MultinomiallogitModel.from_data({'products':args.products, 'etas':list(mnl_etas)}))
        if args.CM in simulator_model_dict.keys():
            choice_model = simulator_model_dict[args.CM]
            features = choice_model.feature()
            data = {'products': args.products}
            for feature in features:
                file_name = 'GT/' + GT_folder + '/Cus_type' + str(i) + '/' + args.CM + '_' + feature + '.npy'
                data[feature] = np.load(file_name).tolist()
            simulator_list.append(choice_model.from_data(data))
        elif args.CM == 'net':
            Gated_net_name = 'GT/' + GT_folder + '/Cus_type' + str(i) + '/AssortNet' + str(args.num_products+1) + '.pth'
            simulator_list.append(torch.load(Gated_net_name))
    if args.CM == 'mnl':
        simulator_list = mnl_list
    if args.CM == 'GT':
        simulator_list = GT_choice_model_list
    mnl_etas_list = np.array(mnl_etas_list) 
    
    ########################################################
    ### train and test
    ########################################################
    if args.detail:
        args.only_test_DRL = True
    from train import train,test,test_DQN
    if args.only_test_Benchmark:
        stream = os.path.join('GT', GT_folder+'/Benchmark_INV'+str(args.ini_inv))
        logger = setup_logger(name='Train', level=20, stream = stream)
        args.logger = logger
        logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
        test(test_sequences, GT_choice_model_list, mnl_etas_list,
             products_price, args, logger, only_test_benchmark=True, only_test_DRL=False)
        logger.info(stream+"completed")
    elif args.only_test_DRL:
        args.load = True
        logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
        if args.load:
            args.stream = 'DRL_save/'+args.folder+args.saved_time_stream
        test(test_sequences, GT_choice_model_list, mnl_etas_list,
             products_price, args, logger, only_test_benchmark=False, only_test_DRL=True)
        logger.info(stream+"completed")
    elif args.only_test_DQN:
        args.load = True
        logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
        if args.load:
            args.stream = 'DQN_save/'+args.folder+args.saved_time_stream
        test_DQN(test_sequences,GT_choice_model_list,mnl_etas_list,initial_inventory,products_price,args,logger,args.load)
        logger.info(stream+"completed")
    elif args.PPO:
        from train import train_PPO
        logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
        best = 0
        round = 1
        if args.training_curves:
            round = 3
        for i in range(round):
            best = train_PPO(best,i,args,simulator_list,mnl_etas_list,products_price,train_sequences,logger)
            logger.info(stream+"completed"+str(i))
        test(test_sequences, GT_choice_model_list, mnl_etas_list,
                 products_price,args,logger,only_test_benchmark=False,only_test_DRL= True)
        logger.info(stream+"completed")
    elif args.PG:
        from train import train_PG
        logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
        best = 0
        round = 1
        if args.training_curves:
            round = 3
        for i in range(round):
            best = train_PG(best,i,args,simulator_list,mnl_etas_list,products_price,train_sequences,logger)
            logger.info(stream+"completed"+str(i))
        test(test_sequences, GT_choice_model_list, mnl_etas_list,
                 products_price,args,logger,only_test_benchmark=False,only_test_DRL= True)
        logger.info(stream+"completed")
    else:
        logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
        best = 0
        round = 1
        if args.training_curves:
            round = 3
        for i in range(round):
            best = train(best,i,args,simulator_list,mnl_etas_list,products_price,train_sequences,logger)
            logger.info(stream+"completed"+str(i))
        if args.DQN:
            args.load = True
            test_DQN(test_sequences,GT_choice_model_list,mnl_etas_list,initial_inventory,products_price,args,logger,args.load)
        else:
            test(test_sequences, GT_choice_model_list, mnl_etas_list,
                 products_price,args,logger,only_test_benchmark=False,only_test_DRL= True)
        logger.info(stream+"completed")
