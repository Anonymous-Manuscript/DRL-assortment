from models.__init__ import Model
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from utils import safe_log
import itertools

class encoder(nn.Module):
    def __init__(self,input_dim,depth,width,output_dim):
        super().__init__()
        fully_conn = []
        if depth==1:
            fully_conn.append(nn.Linear(input_dim, output_dim))
        else:
            fully_conn.append(nn.Linear(input_dim, width))
            fully_conn.append(nn.ReLU())
            for d in range(depth-2):
                fully_conn.append(nn.Linear(width, width))
                fully_conn.append(nn.ReLU())
            fully_conn.append(nn.Linear(width, output_dim))
        self.fully_conn = nn.Sequential(*fully_conn)
    def forward(self,x):
        return self.fully_conn(x)

class Gate_Assort_Net(Model, nn.Module):
    @classmethod#Class methods can be called on the class without creating an instance.
    def code(cls):
        return 'Assort Net'
    @classmethod
    def simple_deterministic(cls, products, width):
        return cls(products, width)

    def __init__(self, products, width):
        Model.__init__(self, products)
        nn.Module.__init__(self)
        input_dim = len(products)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, input_dim)
        )
    def forward(self,x):#应该输入tensor
        score = self.layers(x).mul(x)
        score[score==0]=-1e20
        return score # sotfmax之后是选择各个元素的概率

    def transform_trsanction(self,transaction): # onehot transaction
        X = np.array(transaction)[:, :-1]
        Y = np.array(transaction)[:, -1]
        # 将数据转换成torch形式
        x_train = torch.from_numpy(X)
        x_train = x_train.float()
        y_train = torch.from_numpy(Y)
        y_train = y_train.type(torch.LongTensor)
        return x_train, y_train

    def estimate_from_transaction(self, products, transaction, logger): # onehot transaction
        x_train, y_train = self.transform_trsanction(transaction)
        batch_size = 16
        datasets_train = TensorDataset(x_train, y_train)
        train_iter = DataLoader(datasets_train, batch_size=batch_size, shuffle=True, num_workers=0)
        lossFunc = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=0.001)  # 定义优化器，设置学习率
        epochs = 10  # 训练轮数
        train_loss = []
        print("Training of Assort-Net Begins")
        for e in range(epochs):
            running_loss = 0
            for ass, choice in train_iter:
                optimizer.zero_grad()
                y_hat = self.forward(ass)
                loss = lossFunc(y_hat, choice)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()  ## 将每轮的loss求和
            train_loss.append(running_loss / len(train_iter))
            logger.info("Training Epoch: {}/{} , Training LL: {:.3f} ".format(e + 1, epochs, - running_loss / len(train_iter)))

    def cal_testing_loss(self, transaction):  # onehot transaction
        x_train, y_train = self.transform_trsanction(transaction)
        y_hat = self.forward(x_train)
        lossFunc = nn.CrossEntropyLoss()
        testing_loss = lossFunc(y_hat, y_train).item()
        return testing_loss

    def probability_of(self, transaction): # normal transaction
        ass = torch.zeros(1, len(self.products))
        ass[0, 0] = 1
        ass[0, transaction.offered_products] = 1
        with torch.no_grad():
            prob = torch.softmax(self.forward(ass), 1)
        return prob[0, transaction.product].cpu().numpy()

    def probability_distribution_over(self, ass):
        with torch.no_grad():
            prob = torch.softmax(self.forward(ass), 1)
        return prob


class Gate_Assort_Net_feature(Model, nn.Module):
    @classmethod#Class methods can be called on the class without creating an instance.
    def code(cls):
        return 'Assort Net feature'
    @classmethod
    def simple_deterministic(cls, args, products, width):
        return cls(args, products, width)

    def __init__(self, args, products, width):
        Model.__init__(self, products)
        nn.Module.__init__(self)
        self.args = args
        self.prop_features = args.prop_features
        num_prods_features = num_cus_types = 6
        self.product_encoder = encoder(num_prods_features, 2, self.args.num_products + 1, width)
        self.cus_encoder = encoder(num_cus_types, 2, self.args.num_products + 1, width)
        input_dim = len(products)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, input_dim)
        )
    def forward(self,prod,cus,ass):#应该输入tensor
        e_prod = self.product_encoder(prod)
        e_cust = self.cus_encoder(cus)
        latent_uti = torch.sum(e_prod * e_cust, dim=2)
        input_ = latent_uti.mul(ass)
        score = self.layers(input_).mul(ass)
        score[score == 0] = -1e20
        return score  # sotfmax之后是选择各个元素的概率

    def transform_trsanction(self,sample_list,choose_list): # onehot transaction
        num_samples = len(choose_list)
        sample_list = torch.from_numpy(sample_list)
        sample_list = sample_list.float()
        choose_list = torch.from_numpy(choose_list)
        choose_list = choose_list.reshape(num_samples, 1)
        choose_list = np.repeat(choose_list, self.args.num_products + 1, 1).reshape(num_samples, self.args.num_products + 1, 1)
        choose_list = choose_list.type(torch.LongTensor)
        dataset = TensorDataset(sample_list, choose_list)
        batch_size = 16
        data_iter = DataLoader(dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0)
        return data_iter

    def estimate_from_transaction(self, sample_list,choose_list, logger): # with cus feature
        num_prods_features = num_cus_types = 6
        train_iter = self.transform_trsanction(sample_list,choose_list)
        lossFunc = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(itertools.chain(self.product_encoder.parameters(),
                                                     self.cus_encoder.parameters(),
                                                     self.layers.parameters()), lr=0.001)  # 定义优化器，设置学习率
        epochs = 10  # 训练轮数
        train_loss = []
        print("Training of Assort-Net Begins")
        for e in range(epochs):
            running_loss = 0
            for ass, choice in train_iter:
                optimizer.zero_grad()
                prod = ass[:, :, :num_prods_features]
                cus = ass[:, :, num_prods_features:num_prods_features + num_cus_types] ############
                ass_onehot = ass[:, 0, num_prods_features + num_cus_types:]
                choose = choice[:, 0, 0]

                y_hat = self.forward(prod,cus,ass_onehot)
                loss = lossFunc(y_hat, choose)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()  ## 将每轮的loss求和
            train_loss.append(running_loss / len(train_iter))
            logger.info("Training Epoch: {}/{} , Training LL: {:.3f} ".format(e + 1, epochs, - running_loss / len(train_iter)))

    def cal_testing_loss(self, sample_list,choose_list):  # onehot transaction
        num_prods_features = num_cus_types = 6
        num_samples = len(choose_list)
        sample_list = torch.from_numpy(sample_list)
        sample_list = sample_list.float()
        choose_list = torch.from_numpy(choose_list)
        choose_list = choose_list.reshape(num_samples, 1)
        choose_list = np.repeat(choose_list, self.args.num_products + 1, 1).reshape(num_samples, self.args.num_products + 1, 1)
        choose_list = choose_list.type(torch.LongTensor)

        prod = sample_list[:, :, :num_prods_features]
        cus = sample_list[:, :, num_prods_features:num_prods_features + num_cus_types] ############
        ass_onehot = sample_list[:, 0, num_prods_features + num_cus_types:]
        choose = choose_list[:, 0, 0]

        y_hat = self.forward(prod,cus,ass_onehot)
        
        lossFunc = nn.CrossEntropyLoss()
        testing_loss = lossFunc(y_hat, choose).item()
        return testing_loss

    '''def probability_of(self, transaction_with_cus_feature):
        prod = torch.from_numpy(self.prop_features[transaction_with_cus_feature.product]).reshape(1,-1)
        cus = torch.from_numpy(np.array(transaction_with_cus_feature.cus_feature)).reshape(1,-1)
        ass_onehot = torch.zeros(1, len(self.products))
        ass_onehot[0, 0] = 1
        ass_onehot[0, transaction_with_cus_feature.offered_products] = 1
        with torch.no_grad():
            prob = torch.softmax(self.forward(prod,cus,ass_onehot), 1)
        return prob[0, transaction.product].cpu().numpy()'''

    def probability_distribution_over(self, prod, cus, ass):
        multi = np.zeros((prod.shape[0], prod.shape[1]))
        multi[ass[0].nonzero().ravel()] = 1
        prop_fea = prod * multi
        cus_fea = np.repeat(cus, prod.shape[0], 0)
        ass = ass.repeat(prod.shape[0], 1)
        with torch.no_grad():
            prob = torch.softmax(self.forward(torch.from_numpy(prop_fea).float().reshape(1,prod.shape[0],-1),
                                              cus_fea.float().reshape(1,prod.shape[0],-1),ass.reshape(1,prod.shape[0],-1))[0], 1)[0].reshape(1,-1)
        return prob
