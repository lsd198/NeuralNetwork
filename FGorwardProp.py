from random import random
from random import seed
import numpy as np
import pandas as pd
from math import exp


def initialize_network(data, n_out, n_hidden, n_neuron, split):
    len_row_data = len(data)
    wt_hl = [[[random() for i in range(len(data.columns))] for j in range(n_neuron)] for k in range(n_hidden)]
    # wt_hl = [[random() for i in range(n_neuron + 1)] for i in range(n_hidden)]
    wt_ne = [[random() for i in range(n_neuron)] for i in range(n_out)]
    data.insert(max(data.columns.unique()), max(data.columns.unique()) + 1, [1] * 210)
    target_data = pd.Series.to_list((data[len(data.columns) - 2]))
    data.drop([len(data.columns) - 2], axis='columns', inplace=True)
    dataset = data.copy()
    ttdata = split_traintest(dataset, split)
    sumprod(ttdata[0], wt_hl, n_out, n_hidden, n_neuron)


def split_traintest(dataset, split):
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


def sumprod(tr_data, wt_hl, n_out, n_hidden, n_neuron):
    error = []
    sum_act = []
    falg=0
    for i in range(len(tr_data)):
        ar1 = np.array(list(tr_data.iloc[i]))
        for l in range(n_hidden):
            for k in range(n_neuron):
                ar2 = np.array(wt_hl[l][k])
                arr3 = ar1 * ar2
                sum_act.append(sum(arr3))
            ar1=np.array(sum_act)



def activation():
    print('adsf')


def forward_prop():
    print('dasf')


def backward_propagation():
    print('dasf')


def delta_cal():
    print('adsf')


def wt_update():
    print('asdf')


def fileupload(filename):
    dataset = pd.read_csv(filename, header=None)
    return dataset


seed(2)
data = fileupload('datafile.csv')
data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)
initialize_network(data, n_out=len(data[len(data.columns) - 1].unique()), n_hidden=5, n_neuron=7, split=3)
print('asdfasd')
# below function will initilaize the initial weights for both the hidden neurons  and output layer neurons
