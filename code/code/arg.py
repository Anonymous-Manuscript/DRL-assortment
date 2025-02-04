import argparse


def init_parser(alg):
    if alg == 'A2C':

        parser = argparse.ArgumentParser(description='Thesis')
        parser.add_argument('--logger')
        parser.add_argument('--name')
        parser.add_argument('--gpu', default='0', type=str)
        parser.add_argument('--device')
        # network
        parser.add_argument('--gamma', type=float, default=1, metavar='G', help='discount factor for rewards')
        parser.add_argument('--epsilon_decay', type=float, default=0.998, help='*epsilon')#
        parser.add_argument('--epoch_num', type=int, default=40, help='#training epoches')
        parser.add_argument('--batch_size', default=1, type=int, help='')#1
        parser.add_argument('--layer', type=str, default="128_64", help='Q-Net')
        parser.add_argument('--lr', type=float, default=1e-05, help='learning rate.')
        parser.add_argument('--num_steps', default=10, type=int, help='')
        parser.add_argument('--h', default=2, type=int, help='hidden layer')
        parser.add_argument('--w', nargs='+', default=[64,64], type=int)
        parser.add_argument('--nn_out', default=64, type=int)
        parser.add_argument('--share_lr', type=float, default=0.001, help='share learning rate.')
        parser.add_argument('--actor_lr', type=float, default=0.001, help='actor learning rate.')
        parser.add_argument('--critic_lr', type=float, default=0.001, help='critic learning rate.')
        parser.add_argument('--em_lr', type=float, default=0.001, help='embedding learning rate.')
        parser.add_argument('--step', type=float, default=100, help='learning rate decay step.')
        parser.add_argument('--lr_min', type=float, default=1e-5, help='learning rate minimum.')
        parser.add_argument('--e_rate', type=float, default=0.1, help='coefficient of entropy_loss')
        parser.add_argument('--a_rate', type=float, default=1, help='coefficient of actor_loss')
        parser.add_argument('--c_rate', type=float, default=1, help='coefficient of critic_loss')
        parser.add_argument('--lr_decay_lambda', type=float, default=0.999, help='.') 
        parser.add_argument('--shape', type=bool, default=True, help='.')
        parser.add_argument('--cus_onehot', type=bool, default=False, help='use onehot vector for customer type')
        parser.add_argument('--cus_embedding_dim', type=int, default=4,
                            help='dimension of the embedding vector of customer type')
        parser.add_argument('--t_embedding_dim', type=int, default=1, help='dimension of the embedding vector of time')
        parser.add_argument('--GR_Train', type=bool, default=False, help='Greedy train for DNN')
        parser.add_argument('--GR_Test', type=bool, default=True, help='Greedy test for DNN')
        # aim
        parser.add_argument('--PG', type=bool, default=False, help='whether use policy gradient (False means A2C)')
        parser.add_argument('--PPO', type=bool, default=False, help='whether use PPO (False means A2C)')
        parser.add_argument('--MLP', type=bool, default=False, help='whether use MLP (False means RNN)')
        parser.add_argument('--CM', type=str, default='mc', help='choice model as simulator')
        parser.add_argument('--DQN', type=bool, default=False, help='')
        parser.add_argument('--load', type=bool, default=False, help='')
        parser.add_argument('--only_test_Benchmark', type=bool, default=False, help='')
        parser.add_argument('--only_test_DRL', type=bool, default=False, help='')
        parser.add_argument('--only_test_DQN', type=bool, default=False, help='')
        parser.add_argument('--UB', type=bool, default=False, help='calculate the upper bound of the current ground truth')
        parser.add_argument('--UB_assortment', type=bool, default=False, help='calculate the assortments in upper bound solution')
        parser.add_argument('--training_curves', type=bool, default=True)
        parser.add_argument('--detail', type=bool, default=False)
        # Setting
        parser.add_argument('--GT', default='rl', type=str, help='groundtruth for experiment')
        parser.add_argument('--stream', default='0', type=str, help='name of experiment')
        parser.add_argument('--saved_time_stream', default='2024-12-13-10-40-21', type=str, help='name of saved experiment')
        parser.add_argument('--folder', type=str, default='', help='')
        parser.add_argument('--seed_range', default=10, type=int, help='')
        parser.add_argument('--net_seed', default=0, type=int, help='')
        parser.add_argument('--seed', default=0, type=int, help='')
        parser.add_argument('--info', type=bool, default=True, help='')
        parser.add_argument('--k', type=float, default=0.03, help='')
        parser.add_argument('--L', type=int, default=100, help='mean of horizon length')
        parser.add_argument('--T', type=int, default=500, help='number of horizons')
        parser.add_argument('--cardinality', default=4, type=int, help='size constraint')
        parser.add_argument('--ini_inv', default=10, type=int, help='initial inventory')
        parser.add_argument('--num_products', default=10, type=int, help='')
        parser.add_argument('--products', default=[0,1,2,3,4,5,6,7,8,9,10], type=list, help='')
        parser.add_argument('--num_cus_types', default=4, type=int, help='')
        parser.add_argument('--rank_list', help='array 20*(num_products+1), 20 extracted preference lists')
        parser.add_argument('--cus_type',
                            help='array num_cus_types*20, four prob vectors of four customers, towards 20 extracted preference lists')
        parser.add_argument('--seg_prob')
        parser.add_argument('--seqdata')
        parser.add_argument('--transdata')

        return parser

    else:

        raise RuntimeError('undefined algorithm {}'.format(alg))