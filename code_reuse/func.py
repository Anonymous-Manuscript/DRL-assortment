import numpy as np
import random
import json
import torch

def transaction_train_test_split(onehot_data, data):
    number_samples = len(onehot_data)
    random.seed(0)
    id_list = list(range(number_samples))
    random.shuffle(id_list)
    train_onehot_data = onehot_data[id_list[0: int(number_samples * 0.8)]]
    test_onehot_data = onehot_data[id_list[int(number_samples * 0.8):]]
    train_data = data[id_list[0: int(number_samples * 0.8)]]
    test_data = data[id_list[int(number_samples * 0.8): ]]
    return  train_onehot_data, test_onehot_data, train_data, test_data

# choice model / assortment - related
def prob_of_products(ass,V):
    V = ass * V
    V = np.append(1, V)#the first product is non-purchase
    prob = V/np.sum(V)
    return prob

def exp_rev(ass,V,r):
    V = ass * V
    prob = V/np.sum(V)
    rev = prob@r
    return rev

def Cardinality_ass(guess_V,profits,constraint):
    intersections = []
    pairs = []
    pairs.append((0, 0))
    intersections.append(-999999999)
    # O(N^2)
    for i in range(len(profits)):
        for j in range(i + 1, len(profits)+1):
            # here our count indexing starts from 0, when it should be 1
            pairs.append((i, j))
            if i == 0:
                intersections.append(profits[j-1])#跟横轴的交点
            else:
                numerator = profits[i-1] * guess_V[i-1] - profits[j-1] * guess_V[j-1]
                denominator = guess_V[i-1] - guess_V[j-1]
                intersections.append(numerator / denominator)
    pairs.append((len(profits), len(profits)))
    intersections.append(999999999)
    args = np.argsort(intersections)
    pairs = np.asarray(pairs)[args]

    A = []
    G = set()
    B = set()
    # v deals with only the inside options, drop the outside option
    sigma = np.argsort(-guess_V)  # descending order
    G.update(sigma[:constraint])
    A.append(sigma[:constraint].tolist())
    for i in range(len(pairs)-1):
        if i == 0:
            continue
        if pairs[i][0] != 0:  # last index(column) will be our 0
            # swap order
            swap_values = pairs[i]-1
            swap_index = np.argwhere(np.isin(sigma, swap_values)).flatten()
            if len(swap_index) == 2:
                swap_1, swap_2 = sigma[swap_index[0]], sigma[swap_index[1]]
                sigma[swap_index[0]], sigma[swap_index[1]] = swap_2, swap_1
            else:
                continue
        else:
            B.add(pairs[i][1]-1)
        G = set(sigma[:constraint])
        A_t = G - B
        if A_t:
            A.append(list(A_t))

    profits_ = []
    for assortment in A:
        v = guess_V[assortment]
        w = profits[assortment]
        numerator = np.dot(v, w)
        denominator = 1 + np.sum(v)
        profits_.append(numerator / denominator)

    max_profs_index = np.argmax(profits_)
    ass=A[max_profs_index]

    ass_onehot=np.array([0]*len(profits))
    ass_onehot[ass] = 1
    return ass_onehot

# DRL-related
def compute_returns(next_value, rewards, m_dones, values, gamma=1):
    R = next_value
    returns = []
    returns2 = []
    for step_ in reversed(range(rewards.shape[1])):
        if step_ == rewards.shape[1]-1:
            returns2.append(rewards[:,-1].unsqueeze(dim=1) + gamma * next_value * m_dones[-1])
        else:
            R2 = rewards[:,step_].unsqueeze(dim=1) + \
                gamma * values[:,step_+1] * m_dones[step_]
            returns2.insert(0, R2)
        R = rewards[:,step_].unsqueeze(dim=1) + \
            gamma * R * m_dones[step_]
        returns.insert(0, R)
    return returns, returns2#returns is multi step difference, returns2 is one-step ahead difference


# logger, file-save
import sys
import os
import logging
_streams = {
    "stdout": sys.stdout
}
def setup_logger(name: str, level: int, stream: str = "stdout") -> logging.Logger:
    global _streams
    if stream not in _streams:
        log_folder = os.path.dirname(stream)
        _streams[stream] = open(stream, 'w')
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    for stream in _streams:
        sh = logging.StreamHandler(stream=_streams[stream])
        sh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)