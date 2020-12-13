from random import random
from random import randint
from random import seed
import numpy as np
import pandas as pd
from math import exp

class NeuralNetwork:
    def __init__(self, wt_hl, wt_ne, n_out, n_hidden, n_neuron, split):
        self.wt_hl = wt_hl
        self.wt_ne = wt_ne
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_neuron = n_neuron
        self.split = split
        self.target_data = []


    def initialize_network(self,data):
        len_row_data = len(data)
        self.wt_hl = [[[random() for i in range(len(data.columns))] for j in range(n_neuron)] for k in range(n_hidden)]
        self.wt_ne = [[random() for i in range(n_neuron + 1)] for i in range(n_out)]
        target_data = self.split_traintest(data[len(data.columns) - 1], split)
        error_val = list(data[len(data.columns) - 1].unique())
        error_val.sort()
        data.drop([len(data.columns) - 1], axis='columns', inplace=True)
        ttdata = split_traintest(nor_data(data), split)
        return len_row_data, wt_hl, wt_ne, target_data, error_val, ttdata


    def nor_data(self):
        print()


    def split_traintest(self):
        a = 0
        i = 0
        if int(len(dataset) % split) != 0:
            a = int((dataset % split))
        data_size = int(len(dataset) - a)
        split_size = int(len(dataset) / split)
        from_sp = 0
        split_data = []
        for i in range(split):
            to_sp = split_size * (i + 1)
            split_a = dataset[from_sp:to_sp]
            split_data.append(split_a)
            from_sp = to_sp
        return split_data

    def forward_prop(self):
        print()

    def sep_target(self):
        print()
    def errorcal(self):
        print()
    def backward_propagation(self):
        print()
    def wt_update(self):
        print()

    def fileupload(filename):
        dataset = pd.read_csv(filename, header=None)
        return dataset
seed(1)
data=pd.read_csv('datafile.csv', header=None)
data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)
# Starting of the neural network
forward_prop(data, len(data[len(data.columns) - 1].unique()), 5, 7, 3, 6)
