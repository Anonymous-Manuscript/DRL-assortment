import argparse

def init_parser(alg):

    if alg == 'A2C':

        parser = argparse.ArgumentParser(description='Thesis')
        parser.add_argument('--logger')
        parser.add_argument('--name')
        parser.add_argument('--gpu', default='0', type=str)
        parser.add_argument('--device')

        parser.add_argument('--gamma', type=float, default=1, metavar='G', help='discount factor for rewards (default: 0.99)')
        parser.add_argument('--epsilon_decay', type=float, default=0.998, help='*epsilon')
        parser.add_argument('--epoch_num', type=int, default=40)
        parser.add_argument('--batch_size', default=1, type=int, help='')
        parser.add_argument('--layer', type=str, default="128_64", help='Q-Net')
        parser.add_argument('--lr', type=float, default=0.00001, help='learning rate.')
        parser.add_argument('--num_steps', default=10, type=int, help='')
        parser.add_argument('--h', default=2, type=int, help='hidden layer')
        parser.add_argument('--w', nargs='+', default=[64,64], type=int)
        parser.add_argument('--nn_out', default=64, type=int)
        parser.add_argument('--share_lr', type=float, default=0.01, help='learning rate.')
        parser.add_argument('--actor_lr', type=float, default=0.002, help='learning rate.')
        parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate.')
        parser.add_argument('--em_lr', type=float, default=0.001, help='embedding learning rate.')
        parser.add_argument('--step', type=float, default=20, help='learning rate decay step.')
        parser.add_argument('--lr_min', type=float, default=0.0001, help='learning rate minimum.')
        parser.add_argument('--e_rate', type=float, default=0.1, help='.')
        parser.add_argument('--a_rate', type=float, default=1, help='.')
        parser.add_argument('--c_rate', type=float, default=1, help='.')
        parser.add_argument('--lr_decay_lambda', type=float, default=0.999, help='.')#0.999999999
        parser.add_argument('--shape', type=bool, default=True, help='.')
        parser.add_argument('--cus_onehot', type=bool, default=False, help='use onehot vector for customer type')
        parser.add_argument('--cus_embedding_dim', type=int, default=4,
                            help='dimension of the embedding vector of customer type')
        parser.add_argument('--t_embedding_dim', type=int, default=1, help='dimension of the embedding vector of time')
        parser.add_argument('--GR_Train', type=bool, default=False, help='Greedy train for DNN')
        parser.add_argument('--GR_Test', type=bool, default=True, help='Greedy test for DNN')
        # aim
        parser.add_argument('--MLP', type=bool, default=False, help='whether use MLP (False means RNN)')
        parser.add_argument('--feature', type=bool, default=True, help='whether input customer feature into the net')
        parser.add_argument('--GT', type=str, default='net_feature', help='net-feature or mnl-feature')
        parser.add_argument('--CM', type=str, default='GT', help='choice model as simulator')
        parser.add_argument('--DQN', type=bool, default=False, help='')
        parser.add_argument('--load', type=bool, default=False, help='')
        parser.add_argument('--only_test_Benchmark', type=bool, default=False, help='')
        parser.add_argument('--only_test_DRL', type=bool, default=False, help='')
        parser.add_argument('--only_test_DQN', type=bool, default=False, help='')
        parser.add_argument('--UB', type=bool, default=False, help='calculate the upper bound of the current ground truth')
        parser.add_argument('--training_curves', type=bool, default=True)
        parser.add_argument('--detail', type=bool, default=False)
        # Setting
        parser.add_argument('--stream', default='0', type=str, help='name of experiment')
        parser.add_argument('--saved_time_stream', default='2024-11-07-22-31-10', type=str, help='name of saved experiment')
        parser.add_argument('--folder', type=str, default='', help='')
        parser.add_argument('--seed_range', default=10, type=int, help='')
        parser.add_argument('--net_seed', default=0, type=int, help='')
        parser.add_argument('--seed', default=0, type=int, help='')
        parser.add_argument('--info', type=bool, default=True, help='')
        parser.add_argument('--L', type=int, default=100, help='mean of horizon length')
        parser.add_argument('--cardinality', default=4, type=int, help='size constraint')
        parser.add_argument('--ini_inv', default=5, type=int, help='initial inventory')
        parser.add_argument('--num_products', default=30, type=int, help='')
        parser.add_argument('--products', default=[0,1,2,3,4,5,6,7,8,9,10], type=list, help='')
        parser.add_argument('--prop_features', type=list, help='feature of each product, the first list is the no-purchase option')
        parser.add_argument('--num_cus_types', default=4, type=int, help='')
        parser.add_argument('--num_cus_features', default=6, type=int, help='')
        parser.add_argument('--type_cus_feature', type=dict, help='customer feature lists of each customer type')
        parser.add_argument('--cus_feature_index_list', type=dict, help='in a customer type, which specific customer comes, 211 lists')
        parser.add_argument('--seg_prob')
        parser.add_argument('--seqdata')
        parser.add_argument('--transdata')
        
        return parser

    else:

        raise RuntimeError('undefined algorithm {}'.format(alg))