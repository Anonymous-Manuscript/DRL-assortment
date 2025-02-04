from func import setup_logger
import torch
import numpy as np
from arg import init_parser
import os
import time
import json
import random
from models.ranked_list import RankedListModel
from models.multinomial import MultinomiallogitModel,MultinomiallogitModel_feature
from models.Assort_net import Gate_Assort_Net,Gate_Assort_Net_feature
from models.latent_class import LatentClassModel
from models.mmnl import MixedMNLModel
from models.rcs import RcsModel
from models.exponomial import ExponomialModel
from models.mc import MarkovChainModel
########################################################
###  read args and set logger
########################################################
parser = init_parser('A2C')
args = parser.parse_args()
device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
args.device = device
args.device = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# Assuming args.folder contains the folder name
network = 'RNN'
if args.MLP:
    network = 'MLP'
GT_folder = str(args.num_products)
if args.DQN:
    args.folder = 'N='+str(args.num_products)+'INV='+str(args.ini_inv)+'DQN'+args.CM
else:
    args.folder = 'N='+str(args.num_products)+'INV='+str(args.ini_inv)+network+args.CM
# Create the directory if it doesn't exist
os.makedirs('log/'+args.folder, exist_ok=True)

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
if args.num_products == 100:
    mean = 136
    var = 98
elif args.num_products == 30:
    mean = 156
    var = 72
else:
    mean = 151
    var = 53
products_price = (np.load('GT/'+GT_folder+'/prices.npy')*(var**0.5)+mean)
products_price = products_price/1000

with open('GT/'+GT_folder+'/seqdata_'+str(args.num_products)+'Hotels.json', 'r') as f:
    args.seqdata = json.loads(f.read())
type_cus_feature = {}
for m in range(args.num_cus_types):
    type_cus_feature[m] = np.load('GT/'+GT_folder+'/cue_feature_'+str(args.num_products)+'Hotels_type_'+str(m)+'.npy')
args.type_cus_feature = type_cus_feature
args.prop_features = np.load('GT/'+GT_folder+'/expedia_prop_features.npy')


# initiate ground truth choice model from the loaded parameters
if args.GT == 'net_feature':
    Gated_net_name = 'GT/' + GT_folder + '/AssortNet_feature' + str(args.num_products+1) + '.pth'
    GT_choice_model = torch.load(Gated_net_name)
elif args.GT == 'mnl_feature':
    mnl_feature = MultinomiallogitModel_feature.simple_deterministic(args.products,args.prop_features,len_features=12)
    loaded_etas = np.load('GT/' + GT_folder + '/mnl_feature_etas.npy')
    mnl_feature.update_parameters_from_vector(loaded_etas)
    GT_choice_model = mnl_feature


########################################################
### split train and test
########################################################
cus_feature_indexs = []
for m in range(4):
    cus_feature_indexs.append(args.type_cus_feature[m].shape[0])
args.T = len(args.seqdata.keys())
train_sequences = list(args.seqdata.values())[:int(0.8*args.T)]
seg_prob = np.zeros(args.num_cus_types+1)
train_cus_feature_index_list = []
for l in train_sequences:
    cus_feature_index_l = []
    seg_prob[-1] += len(l)
    for m in range(args.num_cus_types):
        seg_prob[m] += l.count(m)
    for cus_type in l:
        cus_feature_index = np.random.randint(0, cus_feature_indexs[int(cus_type)])
        cus_feature_index_l.append(cus_feature_index)
    train_cus_feature_index_list.append(cus_feature_index_l)
args.seg_prob = (seg_prob/seg_prob[-1])[:-1]
test_sequences = list(args.seqdata.values())[int(0.8*args.T):]
test_cus_feature_index_list = []
for l in test_sequences:
    cus_feature_index_l = []
    for cus_type in l:
        cus_feature_index = np.random.randint(0, cus_feature_indexs[int(cus_type)])
        cus_feature_index_l.append(cus_feature_index)
    test_cus_feature_index_list.append(cus_feature_index_l)

########################################################
### load simulator for each type
########################################################
simulator_model_dict = {'lcmnl':LatentClassModel, 'mc':MarkovChainModel}
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
    simulator_list = GT_choice_model
mnl_etas_list = np.array(mnl_etas_list)
########################################################
### train and test
########################################################
if args.detail:
    args.only_test_DRL = True
from train import train,test,test_DQN
if args.only_test_Benchmark:
    stream = os.path.join('GT', GT_folder+'/Benchmark_INV'+str(args.ini_inv)+args.GT)
    logger = setup_logger(name='Train', level=20, stream = stream)
    args.logger = logger
    logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
    test(test_sequences, test_cus_feature_index_list, GT_choice_model, mnl_etas_list,
         products_price, args, logger, only_test_benchmark=True, only_test_DRL=False)
    logger.info(stream+"completed")
elif args.only_test_DRL:
    args.load = True
    logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
    if args.load:
        args.stream = 'DRL_save/'+args.folder+args.saved_time_stream
    test(test_sequences, test_cus_feature_index_list, GT_choice_model, mnl_etas_list,
         products_price, args, logger, only_test_benchmark=False, only_test_DRL=True)
    logger.info(stream+"completed")
elif args.only_test_DQN:
    args.load = True
    logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
    if args.load:
        args.stream = 'DQN_save/'+args.folder+args.saved_time_stream
    test_DQN(test_sequences,test_cus_feature_index_list, GT_choice_model,mnl_etas_list,initial_inventory,products_price,args,logger,True)
    logger.info(stream+"completed")
else:
    logger.info("Load Factor {}".format(args.L/np.sum(initial_inventory)))
    best = 0
    round = 1
    if args.training_curves:
        round = 3
    for i in range(round):
        best = train(best,i,args,simulator_list,mnl_etas_list,products_price,train_sequences,train_cus_feature_index_list,logger)
        logger.info(stream+"completed"+str(i))
    if args.DQN:
            test_DQN(test_sequences,test_cus_feature_index_list, GT_choice_model,mnl_etas_list,initial_inventory,products_price,args,logger,True)
    else:
        test(test_sequences, test_cus_feature_index_list, GT_choice_model, mnl_etas_list,
             products_price,args,logger,only_test_benchmark=False,only_test_DRL= True)
    logger.info(stream+"completed")
