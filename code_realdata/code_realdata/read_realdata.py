import numpy as np
import random
import json
import os
import torch
import time
from models.ranked_list import RankedListModel
from models.Assort_net import Gate_Assort_Net,Gate_Assort_Net_feature
from models.multinomial import MultinomiallogitModel,MultinomiallogitModel_feature
from models.latent_class import LatentClassModel
from models.rcs import RcsModel
from models.mmnl import MixedMNLModel
from models.exponomial import ExponomialModel
from models.shadow_exp2 import S_EXP2
from models.mc import MarkovChainModel
from estimation.mc_estimate import MarkovChainExpectationMaximizationEstimator
from estimation.rl_estimate import RankedListExpectationMaximizationEstimator
from estimation.ranked_list_explore import MIPMarketExplorer
from estimation.lcmnl_estimate import LatentClassExpectationMaximizationEstimator
from estimation.mc_estimate import ExpectationMaximizationEstimator
from estimation.optimization import Settings, set_Settings
import pandas as pd
from arg import init_parser
from GT.transactions_arrival import Transaction,data_for_estimtion
from func import transaction_train_test_split, NpEncoder, setup_logger
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
#########################################################################################
# initialize args, ground truth, save
#########################################################################################
parser = init_parser('A2C')
args = parser.parse_args()
# Assuming args.folder contains the folder name
args.folder = str(args.num_products)
# Create the directory if it doesn't exist
os.makedirs('GT/'+args.folder, exist_ok=True)

stream = os.path.join('GT', args.folder+'/SimulationLog')
logger = setup_logger(name='Train', level=20, stream = stream)
args.logger = logger
logger.info(args)
logger.info(stream)

np.random.seed(0)
random.seed(0)

#########################################################################################
# read and transform transaction data
#########################################################################################
data = pd.read_csv('GT/'+args.folder+'/trans_data_'+str(args.num_products)+'Hotels.csv',index_col=0)
num_prods_features = 6
num_cus_types = 4
prop_features = [np.zeros(num_prods_features)]
prices = []
for prop in range(args.num_products):
    features = data[data['prop_id']==prop].iloc[0,3:9].values
    prop_features.append(features)
    prices.append(features[-1])
args.prop_features = prop_features
prop_features = np.array(prop_features)#(31, 6)shape
np.save('GT/'+args.folder+'/expedia_prop_features.npy',prop_features)
np.save('GT/'+args.folder+'/prices.npy',np.array(prices))
train_data,test_data = data_for_estimtion(data,prop_features)
#breakpoint()
transactions,transactions_with_cus_feature,_,sample_list,choose_list,transdata_onehot,_ = train_data

#########################################################################################
# initiate and estimate choice models
#########################################################################################
products = [i for i in range(args.num_products+1)]
width = args.num_products + 1

mnl = MultinomiallogitModel.simple_deterministic(products)
mnl_feature = MultinomiallogitModel_feature.simple_deterministic(products,prop_features,len_features=12)
lcmnl = LatentClassModel.simple_deterministic(products, 4)
mc = MarkovChainModel.simple_deterministic(products)
assort_net = Gate_Assort_Net.simple_deterministic(products, width=width)
assort_net_feature = Gate_Assort_Net_feature.simple_deterministic(args, products, width=width)
#################################### train
data, data_with_cus_feature = np.array(transactions), np.array(transactions_with_cus_feature)
# 1
start_time = time.time()
mnl.estimate_from_transaction(products, data)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info("elapsed_time for mnl:{}".format(elapsed_time))
start_time = time.time()
mnl_feature.estimate_from_transaction(products, data_with_cus_feature)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info("elapsed_time for mnl_feature:{}".format(elapsed_time))
# 3
start_time = time.time()
Settings = set_Settings(LATENT_CLASS_SETTINGS)
lcmnl_estimator = LatentClassExpectationMaximizationEstimator()
lcmnl = lcmnl_estimator.estimate(lcmnl, data)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info("elapsed_time for lcmnl:{}".format(elapsed_time))
# 4
start_time = time.time()
Settings = set_Settings(MC_SETTINGS)
mc_estimator = MarkovChainExpectationMaximizationEstimator()
mc = mc_estimator.estimate(mc, data)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info("elapsed_time for mc:{}".format(elapsed_time))
# 5
start_time = time.time()
assort_net.estimate_from_transaction(products, np.array(transdata_onehot), logger)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info("elapsed_time for assort_net:{}".format(elapsed_time))
start_time = time.time()
assort_net_feature.estimate_from_transaction(sample_list, choose_list, logger)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info("elapsed_time for assort_net_feature:{}".format(elapsed_time))
#In-Sample
logger.info("number of samples: {}".format(
    len(transactions)))
logger.info("Overall In-Sample Testing LL - mnl: {:.3f}, mnl_feature: {:.3f}, lcmnl: {:.3f}, mc: {:.3f}, assort_net: {:.3f}, assort_net_feature: {:.3f}.".format(
    mnl.log_likelihood_for(np.array(transactions)),
    mnl_feature.log_likelihood_for(np.array(transactions_with_cus_feature)),
    lcmnl.log_likelihood_for(np.array(transactions)),
    mc.log_likelihood_for(np.array(transactions)),
    -assort_net.cal_testing_loss(np.array(transdata_onehot)),
    -assort_net_feature.cal_testing_loss(sample_list,choose_list)))
#################################### test
transactions,transactions_with_cus_feature,_,sample_list,choose_list,transdata_onehot,_ = test_data
logger.info("number of samples: {}".format(
    len(transactions)))
logger.info("Overall Testing LL - mnl: {:.3f}, mnl_feature: {:.3f}, lcmnl: {:.3f}, mc: {:.3f}, assort_net: {:.3f}, assort_net_feature: {:.3f}.".format(
    mnl.log_likelihood_for(np.array(transactions)),
    mnl_feature.log_likelihood_for(np.array(transactions_with_cus_feature)),
    lcmnl.log_likelihood_for(np.array(transactions)),
    mc.log_likelihood_for(np.array(transactions)),
    -assort_net.cal_testing_loss(np.array(transdata_onehot)),
    -assort_net_feature.cal_testing_loss(sample_list,choose_list)))

#################################### save
'''
for choice_model, cm_name in zip([mnl, mnl_feature, lcmnl, mc],['mnl', 'mnl_feature', 'lcmnl', 'mc']):
    choice_model_data = choice_model.data()
    for key in choice_model_data.keys():
        file_name = 'GT/'+args.folder+'/'+cm_name+'_'+key+'.npy'
        np.save(file_name, np.array(choice_model_data[key]))
file_name = 'GT/'+args.folder+'/AssortNet'+str(width)+'.pth'
torch.save(assort_net, file_name)
file_name = 'GT/'+args.folder+'/AssortNet_feature'+str(width)+'.pth'
torch.save(assort_net_feature, file_name)'''

for i in range(4):
    os.makedirs('GT/'+args.folder+'/Cus_type'+str(i), exist_ok=True)
    logger.info("customer type: "+str(i))
    # 1
    mnl = MultinomiallogitModel.simple_deterministic(products)
    # 2
    lcmnl = LatentClassModel.simple_deterministic(products, 4)
    # 4
    mc = MarkovChainModel.simple_deterministic(products)
    # 5
    assort_net = Gate_Assort_Net.simple_deterministic(products, width=width, type=True)
    ##################################################
    ##################### estimate each choice model for each customer type
    ##################################################
    # train
    transactions, transactions_with_cus_feature, type_transactions, sample_list, choose_list, transdata_onehot, custype_transdata_onehot = train_data
    data = np.array(type_transactions[str(i)])
    data_onehot = np.array(custype_transdata_onehot[str(i)])
    # 1
    start_time = time.time()
    mnl.estimate_from_transaction(products, data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time for mnl:{}".format(elapsed_time))
    # 3
    start_time = time.time()
    Settings = set_Settings(LATENT_CLASS_SETTINGS)
    lcmnl_estimator = LatentClassExpectationMaximizationEstimator()
    lcmnl = lcmnl_estimator.estimate(lcmnl, data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time for lcmnl:{}".format(elapsed_time))
    # 4
    start_time = time.time()
    Settings = set_Settings(MC_SETTINGS)
    mc_estimator = MarkovChainExpectationMaximizationEstimator()
    mc = mc_estimator.estimate(mc, data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time for mc:{}".format(elapsed_time))
    # 5
    start_time = time.time()
    assort_net.estimate_from_transaction(products, data_onehot, logger)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("elapsed_time for assort_net:{}".format(elapsed_time))
    #In-Sample
    logger.info("number of samples: {}".format(
        len(type_transactions[str(i)])))
    data = np.array(type_transactions[str(i)])
    data_onehot = np.array(custype_transdata_onehot[str(i)])
    logger.info("number of samples: {}".format(
        len(type_transactions[str(i)])))
    logger.info("Type In-Sample Testing LL - mnl: {:.3f}, lcmnl: {:.3f}, mc: {:.3f}, assort_net: {:.3f}.".format(
        mnl.log_likelihood_for(data),
        lcmnl.log_likelihood_for(data),
        mc.log_likelihood_for(data),
        -assort_net.cal_testing_loss(data_onehot)))
    #################################### test
    _,_,type_transactions,_,_,_,custype_transdata_onehot = test_data
    data = np.array(type_transactions[str(i)])
    data_onehot = np.array(custype_transdata_onehot[str(i)])
    logger.info("number of samples: {}".format(
        len(type_transactions[str(i)])))
    logger.info("Type Testing LL - mnl: {:.3f}, lcmnl: {:.3f}, mc: {:.3f}, assort_net: {:.3f}.".format(
        mnl.log_likelihood_for(data),
        lcmnl.log_likelihood_for(data),
        mc.log_likelihood_for(data),
        -assort_net.cal_testing_loss(data_onehot)))
    #################################### save
    '''
    for choice_model, cm_name in zip([mnl, lcmnl, mc],['mnl', 'lcmnl', 'mc']):
        choice_model_data = choice_model.data()
        for key in choice_model_data.keys():
            file_name = 'GT/'+args.folder+'/Cus_type'+str(i)+'/'+cm_name+'_'+key+'.npy'
            np.save(file_name, np.array(choice_model_data[key]))
    file_name = 'GT/'+args.folder+'/Cus_type'+str(i)+'/AssortNet'+str(width)+'.pth'
    torch.save(assort_net, file_name)'''



'''
torch.save(product_encoder, 'resnet/ex_product_encoder.pth')
torch.save(cus_encoder, 'resnet/ex_cus_encoder.pth')
torch.save(net, 'resnet/ex_net.pth')'''




