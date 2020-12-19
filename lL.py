import pandas as pd
import numpy as np
from random import  seed
class NeuralNet:
        def __init__(self, epochs, hidden, split_size):
            self.net_para = [hidden, epochs,split_size]
            self.output_ = []
            self.output_nodes = 0
            self.target_data = []
            self.weight = []
            self.output = 0

        def init_weight(self):
            seed(121)
            wt = [for i in range()]



        def forward_prop(self):
            print()


        def activation_fun(self):
            print()


        def back_prop(self):
            print()


        def error(self):
            print()


        def normalize_data(self,dataset):
            for i in range(len(dataset.columns)):
                print(i)
                dataset[i] = (dataset[i]-min(dataset[i]))/(max(dataset[i])-min(dataset[i]))
            return dataset


        def train_test(self,dataset):
           if self.split_size != 1:
               print('Hello split required')


        def start(self, filename):
            dataset = pd.read_csv(filename, header=None)
            self.target_data = dataset[len(dataset.columns)-1]
            dataset = self.normalize_data(dataset.iloc[:, :len(dataset.columns)-1])
            self.train_test(dataset)
            print(dataset)
            self.init_weight()


neural = NeuralNet(5,1000,1)
neural.start('datafile.csv')
