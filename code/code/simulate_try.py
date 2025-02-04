import numpy as np
import random
import json
import os
import torch
import time
from models.ranked_list import RankedListModel
from models.Assort_net import Gate_Assort_Net
from models.multinomial import MultinomiallogitModel
from models.latent_class import LatentClassModel
from models.rcs import RcsModel
from models.mmnl import MixedMNLModel
from models.exponomial import ExponomialModel
from models.shadow_exp2 import S_EXP2
from models.mc import MarkovChainModel
from estimation.mc_estimate import MarkovChainExpectationMaximizationEstimator
from estimation.rl_estimate import RankedListExpectationMaximizationEstimator
from estimation.ranked_list_explore import MIPMarketExplorer
from estimation.lcmnl_estimate import LatentClassExpectationMaximizationEstimator,LatentClassFrankWolfeEstimator,FrankWolfeMNLMixEst
from estimation.mc_estimate import ExpectationMaximizationEstimator
from estimation.optimization import Settings, set_Settings

MC_SETTINGS = {
    'linear_solver_partial_time_limit': 300,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 1800.0,
}
LATENT_CLASS_SETTINGS = {
    'linear_solver_partial_time_limit': None,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 1800.0,
}

from GT.transactions_arrival import TransactionGenerator
from arg import init_parser
from func import transaction_train_test_split, NpEncoder, setup_logger

#########################################################################################
# initialize args, ground truth, save
#########################################################################################
parser = init_parser('A2C')
args = parser.parse_args()
# Assuming args.folder contains the folder name
args.folder = args.GT+'N='+str(args.num_products)+'M='+str(args.num_cus_types)+'k='+str(args.k)+'L='+str(args.L)+'T='+str(args.T)
# Create the directory if it doesn't exist
os.makedirs('GT/'+args.folder, exist_ok=True)

stream = os.path.join('GT', args.folder+'/SimulationLog_try')
logger = setup_logger(name='Train', level=20, stream = stream)
args.logger = logger
logger.info(args)
logger.info(stream)

np.random.seed(0)
random.seed(0)

N = args.num_products
products = [i for i in range(N+1)]
M = args.num_cus_types
#generate prices
prices = np.random.uniform(10,25,N)
prices = np.sort(prices)
logger.info('prices:{}'.format(prices))
np.save('GT/'+args.folder+'/prices.npy', prices)

if args.GT == 'rl':
    groundtruth_list = RankedListModel.initialize_MultiType_groundtruth(products, M, args.folder) # a list, includes M rl Models (M types)
elif args.GT == 'lcmnl':
    groundtruth_list = LatentClassModel.initialize_MultiType_groundtruth(products, M, args.folder) # a list, includes M lcmnl Models (M types)
elif args.GT == 'mmnl':
    groundtruth_list = MixedMNLModel.initialize_MultiType_groundtruth(products, M, args.folder) # a list, includes M mmnl Models (M types)
#########################################################################################
# generate transaction data and arrival data based on ground truth, save
#########################################################################################
# transaction is a dictionary: {customer type 0: [transaction of type 0]}
# where transaction is of type Transaction in transaction_arrival
if args.GT == 'rl':
    custype_transdata_onehot, transaction, arrival = TransactionGenerator(groundtruth_list).gene_MultiType_rl_data(args)
elif args.GT == 'lcmnl':
    custype_transdata_onehot, transaction, arrival = TransactionGenerator(groundtruth_list).gene_lcmnl_data(args)
elif args.GT == 'mmnl':
    custype_transdata_onehot, transaction, arrival = TransactionGenerator(groundtruth_list).gene_mmnl_data(args)
transdata = json.dumps(custype_transdata_onehot, cls=NpEncoder)
seqdata = json.dumps(arrival, cls=NpEncoder)
with open('GT/' + args.folder + '/transdata.json', 'w') as json_file:
    json_file.write(transdata)
with open('GT/' + args.folder + '/seqdata.json', 'w') as json_file:
    json_file.write(seqdata)

#########################################################################################
# estimate choice models for each type, save
#########################################################################################
for i in range(M):
    os.makedirs('GT/'+args.folder+'/Cus_type'+str(i), exist_ok=True)
    logger.info("customer type: "+str(i))
    ##################################################
    ##################### initialize choice models
    ##################################################
    # 1
    mnl = MultinomiallogitModel.simple_deterministic(products)
    # 2
    lcmnl = LatentClassModel.simple_deterministic(products, 20)
    # 3
    rcs = RcsModel.simple_detetministic(products)
    #rl = RankedListModel.simple_deterministic_independent(products)
    #exp = S_EXP2.simple_deterministic(products)
    # 4
    mc = MarkovChainModel.simple_deterministic(products)
    # 5
    width=N+1
    assort_net = Gate_Assort_Net.simple_deterministic(products, width=width)
    ##################################################
    ##################### estimate each choice model for each customer type
    ##################################################
    onehot_data, data = np.array(custype_transdata_onehot[str(i)]), np.array(transaction[str(i)])
    ##################### split
    train_onehot_data, test_onehot_data, train_data, test_data = transaction_train_test_split(onehot_data, data)##############
    ##################### train
    
    # 1
    start_time = time.time()
    mnl.estimate_from_transaction(products, train_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time for mnl:{}".format(elapsed_time))
    print('mnl loss:',len(train_data)*mnl.log_likelihood_for(train_data))
    
    # 3
    start_time = time.time()
    #Settings = set_Settings(LATENT_CLASS_SETTINGS)
    #lcmnl_estimator = LatentClassExpectationMaximizationEstimator()
    #lcmnl = lcmnl_estimator.estimate(lcmnl, train_data)
    mixEst = FrankWolfeMNLMixEst(args)
    mixEst.fit_to_choice_data(train_data, num_iters=20)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time for lcmnl:{}".format(elapsed_time))
    lcmnl.update_parameters_from_vector(np.append(mixEst.mix_props,mixEst.coefs_[:,1:].ravel()))
    print(lcmnl.data())
    #breakpoint()
    
    




    
    
    