from random import random
from random import seed
import numpy as np
import pandas as pd
from math import exp


def initialize_network(data, n_out, n_hidden, n_neuron, split):
    len_row_data = len(data)
    wt_hl = [[[random() for i in range(len(data.columns) - 1)] for j in range(n_neuron)] for k in range(n_hidden)]
    wt_ne = [[random() for i in range(n_neuron)] for i in range(n_out)]
    target_data = data[len(data.columns)-1]
    data.drop([len(data.columns) - 1], axis='columns', inplace=True)
    ttdata = split_traintest(data, split)
    sumprod(nor_data(ttdata[0]), wt_hl, wt_ne, n_out, n_hidden, n_neuron)


def nor_data(dataset):
    newdata=pd.DataFrame()
    for row in dataset:
        newdata.insert(row, row, (dataset[row] - dataset[row].max())/ (dataset[row].max() - dataset[row].min()))
    return newdata

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


def sumprod(tr_data, wt_hl, wt_ne, n_out, n_hidden, n_neuron):
    error = []
    sum_act = []
    flag = 0
    for i in range(len(tr_data)):
        ar1 = np.array(list(tr_data.iloc[i]))
        for l in range(n_hidden + 1):
            if flag <= (n_hidden - 1):
                sum_act.clear()
                for k in range(n_neuron):
                    ar2 = np.array(wt_hl[l][k])
                    arr3 = ar1 * ar2
                    sum_act.append(sum(arr3))
                ar1 = np.array(sum_act)
                flag = flag + 1
            else:
                sum_act.clear()
                for m in range(len(wt_ne)):
                    ar2 = np.array(wt_ne[i])
                    arr3 = sum(ar1 * ar2)
                    arr4 = 1 / (1 + exp(-arr3))
                    sum_act.append(arr3)
                errorcal(sum_act)


def sep_target(ttdata):
    tg=[]
    for val in ttdata:
        for row in val:
            if row == len(val.columns)-1:
                tg.append(val[row])
    return tg
def errorcal(arr4):
    print('')

def remove_target(ttdata):
    print('')


def backward_propagation():
    print('dasf')


def delta_cal():
    print('adsf')


def wt_update():
    print('asdf')


def fileupload(filename):
    dataset = pd.read_csv(filename, header=None)
    return dataset


seed(121)
data = fileupload('datafile.csv')
data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)
initialize_network(data, n_out=len(data[len(data.columns) - 1].unique()), n_hidden=5, n_neuron=7, split=3)
print('asdfasd')
# below function will initilaize the initial weights for both the hidden neurons  and output layer neurons
