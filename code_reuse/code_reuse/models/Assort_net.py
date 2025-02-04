from models.__init__ import Model
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from utils import safe_log

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
