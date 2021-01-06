import pandas as pd
import numpy as np
from random import seed


class NeuralNet:
    def __init__(self, hidden, nodes, epochs, split_size):
        self.net_para = [hidden, nodes, epochs, split_size]
        self.layer_output = []
        self.output_nodes = 0
        self.net_para2 = []
        self.weight = []
        self.output = 0

    def init_weight(self):
        for i in range(self.net_para[0] + 1):
            if i == 0:
                mat_temp = np.random.rand(self.net_para2[1] + 1, self.net_para[1])
                # self.weight.append(np.vstack((mat_temp, np.array([1]*self.net_para[1]))))
            elif i == self.net_para[0]:
                mat_temp = np.random.rand(self.net_para[1] + 1, self.net_para2[2])
                # self.weight.append(np.vstack((mat_temp, np.array([1] * self.net_para2[2]))))
            else:
                mat_temp = np.random.rand(self.net_para[1] + 1, self.net_para[1])
                # self.weight.append(np.vstack((mat_temp, np.array([1] * self.net_para[1]))))
            self.weight.append(mat_temp)

    def forward_prop(self, ip):
        self.ip = ip
        for layers in range(self.net_para[0] + 1):
            input = np.array(np.dot((self.ip.reshape(1, int(list(self.weight[0].shape)[0]))), self.weight[0]))
            self.ip = input
            self.layer_output.append(input)

    def activation_fun(self):
        print()

    def back_prop(self):
        print()

    def error(self):
        print()

    def normalize_data(self, dataset):
        for i in range(len(dataset.columns)):
            dataset[i] = (dataset[i] - min(dataset[i])) / (max(dataset[i]) - min(dataset[i]))
        return dataset

    def train_test(self, dataset):
        if self.net_para[3] != 1:
            print('Hello split required')

    def start(self, filename):
        dataset = pd.read_csv(filename, header=None)
        self.net_para2 = [dataset[len(dataset.columns) - 1], len(dataset.columns) - 1,
                          len(dataset[len(dataset.columns) - 1].unique())]
        dataset = self.normalize_data(dataset.iloc[:, :len(dataset.columns) - 1])
        self.train_test(dataset)
        print(dataset)
        self.init_weight()
        print(self.weight)
        # dataset[len(dataset.columns)] = [1] * len(dataset)
        for epochs in range(self.net_para[2]):
            for ip in range(len(dataset)):
                self.forward_prop(np.array(dataset.iloc[ip]))

# testing the git hub
neural = NeuralNet(5, 2, 1000, 1)
neural.start('datafile.csv')
