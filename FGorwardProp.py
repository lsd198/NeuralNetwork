from random import random
from random import seed
import numpy as np
import pandas as pd
from math import exp


def initialize_network(data, n_out, n_hidden, n_neuron, split):
    len_row_data = len(data)
    np.random.seed(121)
    wt_hl = [[[random() for i in range(len(data.columns))] for j in range(n_neuron)] for k in range(n_hidden)]
    wt_ne = [[random() for i in range(n_neuron + 1)] for i in range(n_out)]
    target_data = split_traintest(data[len(data.columns) - 1], split)
    error_val = list(data[len(data.columns) - 1].unique())
    error_val.sort()
    data.drop([len(data.columns) - 1], axis='columns', inplace=True)
    ttdata = split_traintest(nor_data(data), split)
    sumprod(ttdata[0], wt_hl, wt_ne, n_out, n_hidden, n_neuron, error_val,target_data[0])


def nor_data(dataset):
    newdata = pd.DataFrame()
    for row in dataset:
        newdata.insert(row, row, (dataset[row] - dataset[row].min()) / (dataset[row].max() - dataset[row].min()))
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


def sumprod(tr_data, wt_hl, wt_ne, n_out, n_hidden, n_neuron, error_val,target_data):
    # error = pd.DataFrame(,)
    err_final=[]
    sum_act = []
    flag = 0
    # Belo loop will run for total len of dataset
    for i in range(len(tr_data)):
        ar1 = np.array(list(tr_data.iloc[i]))
        # Below loop will run for total number of hidden layer that we have
        for l in range(n_hidden + 1):
            if flag <= (n_hidden - 1):
                sum_act.clear()
                # Below code will run for total number of neuron that we have in the hidden layer
                for k in range(n_neuron):
                    ar2 = np.array(wt_hl[l][k])
                    arr3 = ar1 * ar2[0:len(ar2) - 1]
                    sum_act.append(sum(arr3) + ar2[-1])
                ar1 = np.array(sum_act)
                flag = flag + 1
            else:
                # This else part will run only for the final layer that we have  in the network
                sum_act.clear()
                output = []
                for m in range(len(wt_ne)):
                    ar2 = np.array(wt_ne[m])
                    arr3 = sum(ar1 * ar2[len(ar2) - 1]) + ar2[-1]
                    output.append(1 / (1 + exp(-arr3)))
                flag = 0
                errorcal(i, output, error_val,target_data,err_final)


def sep_target(ttdata):
    tg = []
    for val in ttdata:
        for row in val:
            if row == len(val.columns) - 1:
                tg.append(val[row])
    return tg


def errorcal(i, output, error_val,target_data,err_final):
    e_list=[]
    diff_list=[]
    for val in error_val:
        if val == target_data[i]:
            e_list.append(1)
        else:
            e_list.append(0)
    for l1,l2 in zip(e_list,output):
        diff_list.append((l1-l2)*(l1-l2))
    err_final.append(sum(diff_list))









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



data = fileupload('datafile.csv')
data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)
initialize_network(data, n_out=len(data[len(data.columns) - 1].unique()), n_hidden=5, n_neuron=7, split=3)
print('asdfasd')
# below function will initilaize the initial weights for both the hidden neurons  and output layer neurons
