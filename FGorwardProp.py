from random import random
from random import randint
from random import seed
import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt

err_total = []
pred_output = []


# Network initialization
def initialize_network(data, n_out, n_hidden, n_neuron, split):
    len_row_data = len(data)
    wt_hl = [[[random() for i in range(len(data.columns))] for j in range(n_neuron)] for k in range(n_hidden)]
    wt_ne = [[random() for i in range(n_neuron + 1)] for i in range(n_out)]
    target_data = split_traintest(data[len(data.columns) - 1], split)
    error_val = list(data[len(data.columns) - 1].unique())
    error_val.sort()
    data.drop([len(data.columns) - 1], axis='columns', inplace=True)
    ttdata = split_traintest(nor_data(data), split)
    return len_row_data, wt_hl, wt_ne, target_data, error_val, ttdata


# Normalization of the data set
def nor_data(dataset):
    newdata = pd.DataFrame()
    for row in dataset:
        newdata.insert(row, row, (dataset[row] - dataset[row].min()) / (dataset[row].max() - dataset[row].min()))
    return newdata


# splitting the data set into test and train
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


# Forward propagation of the weights
def forward_prop(data, out, hidden, neuron, split, epochs):
    output_all_layer = []
    err_op_neuron = []

    sum_act = []
    flag = 0
    init_op = initialize_network(data, out, hidden, neuron, split)
    tr_data = init_op[5][randint(1, split)-1]
    wt_hl = init_op[1]
    wt_ne = init_op[2]
    n_hidden = hidden
    n_neuron = neuron
    error_val = init_op[4]
    target_data = init_op[3][0]
    train_dat = randint(1, 3)
    for epc in range(epochs):
        accuracy_matrix(epc)
        for i in range(len(tr_data)):
            ar1 = np.array(list(tr_data.iloc[i]))
            # Below loop will run for total number of hidden layer that we have
            for l in range(n_hidden + 1):
                if flag <= (n_hidden - 1):
                    sum_act.clear()
                    # Below code will run for total number of neuron that we have in the hidden layer
                    for k in range(n_neuron):
                        ar2 = np.array(wt_hl[l][k])
                        arr3 = 1 / (1 + exp(-(sum((ar1 * ar2[0:len(ar2) - 1])) + ar2[-1])))
                        sum_act.append(arr3)
                    ar1 = np.array(sum_act)
                    output_all_layer.append(list(ar1))
                    flag = flag + 1

                else:
                    # This else part will run only for the final layer that we have  in the network
                    sum_act.clear()
                    output_final_layer = []
                    for m in range(len(wt_ne)):
                        ar2 = np.array(wt_ne[m])
                        arr3 = sum(ar1 * ar2[len(ar2) - 1]) + ar2[-1]
                        output_final_layer.append(1 / (1 + exp(-arr3)))
                    flag = 0
                    errorcal(i, output_final_layer, error_val, target_data, err_total, err_op_neuron)

            op_back = backward_propagation(wt_hl, wt_ne, err_op_neuron, output_final_layer, output_all_layer)
            updated_wt = wt_update(op_back, wt_hl, wt_ne, n_hidden, 0.30)
            wt_hl = updated_wt[1]
            wt_ne = updated_wt[0]
            output_all_layer.clear()
            print(err_total)
            err_total.clear()
            output_final_layer.clear()
            err_op_neuron.clear()
    return err_total


# /separating the targets
def sep_target(ttdata):
    tg = []
    for val in ttdata:
        for row in val:
            if row == len(val.columns) - 1:
                tg.append(val[row])
    return tg


# Error Calculations and stirubnf the weights in the variables
def errorcal(i, output, error_val, target_data, err_total, err_op_neuron):
    e_list = []
    diff_list = []
    for val in error_val:
        if val == target_data[i]:
            e_list.append(1)
        else:
            e_list.append(0)
    for l1, l2 in zip(e_list, output):
        diff_list.append((l1 - l2) * (l1 - l2))
    err_op_neuron.append(diff_list)
    pred_output.append((output.index(max(output))) + 1)
    err_total.append(sum(diff_list))


def accuracy_matrix(i):
    if i != 0:
        print(pred_output)


# back Propagation of the weights
def backward_propagation(wt_hl, wt_ne, err_op_neuron, output_final_layer, output_all_layer):
    temp_ne = []
    temp_hl = []
    temp = []
    delta_all_h_layer = np.array([[a * (1 - a) for a in i] for i in output_all_layer])
    output_all_layer_conv = np.array([i for i in output_all_layer])
    wt_ne_conv = np.array([i for i in wt_ne])
    wt_hl_conv = np.array([i for i in wt_hl])
    delta_neuron = (
            np.array(err_op_neuron) * np.array(output_final_layer) * (1 - np.array(output_final_layer))).reshape(
        len(output_final_layer), )
    temp_val = 0
    # Loop will run  number of times we have the layer
    for h_layer in reversed(range(len(wt_hl))):
        # For loop will run number of times we have the nodes
        for node in range(len(wt_hl[0])):
            # For loop will run number of time weights in particular node
            for weight in range(len(wt_hl[0][node])):
                if h_layer == 4:
                    if temp_val < 7:
                        temp.append(
                            sum(wt_ne_conv[:, node] * delta_neuron) * (output_all_layer_conv[h_layer - 1][weight]) * (
                                delta_all_h_layer[h_layer][node]))
                        temp_val = temp_val + 1

                    else:
                        temp.append(
                            sum(wt_ne_conv[:, node] * delta_neuron) * 1 * (
                                delta_all_h_layer[h_layer][node]))
                        temp_val = 0
                else:
                    if temp_val < 7:
                        temp.append(
                            sum(wt_hl_conv[h_layer + 1][:, node] * delta_all_h_layer[h_layer + 1]) * (
                                output_all_layer_conv[h_layer - 1][weight]) * (delta_all_h_layer[h_layer][node]))
                        temp_val = temp_val + 1
                    else:
                        temp.append(sum(wt_hl_conv[h_layer + 1][:, node] * delta_all_h_layer[h_layer + 1]) * (
                                1 * (delta_all_h_layer[h_layer][node])))
                        temp_val = 0
            temp_hl.append(temp.copy())
            temp.clear()

    for i in range(len(wt_ne)):
        for j in range(len(wt_ne[i])):
            if j == (len(wt_ne[i]) - 1):
                temp.append(delta_neuron[i])
            else:
                temp.append(delta_neuron[i] * output_all_layer_conv[len(wt_hl) - 1][j])
        temp_ne.append(temp.copy())
        temp.clear()

    return temp_hl, temp_ne


# Updating the weights
def wt_update(temp_wt, wt_hl, wt_ne, n_hidden, lr):
    temp_ne = np.array([i for i in temp_wt[1]])
    temp_hl = np.array([temp_wt[0][i:i + 7] for i in range(0, len(temp_wt[0]), 7)])
    # temp_hl = np.array([np.column_stack((i, [1] * 7)) for i in temp_hl])
    wt_hl = np.array([i for i in wt_hl])
    wt_ne = np.array([i for i in wt_ne])
    wt_ne = (wt_ne - lr * temp_ne).tolist()
    wt_hl = (wt_hl - lr * temp_hl).tolist()
    return wt_ne, wt_hl


# Reading the data set for the neural network
def fileupload(filename):
    dataset = pd.read_csv(filename, header=None)
    return dataset


seed(1)
data = fileupload('datafile.csv')
data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)
# Starting of the neural network
forward_prop(data, len(data[len(data.columns) - 1].unique()), 5, 7, 1, 4)
# a = [i for i in range(0, 500, 1)]
# below function will initilaize the initial weights for both the hidden neurons  and output layer neurons
