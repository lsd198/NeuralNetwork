from random import random
from random import seed
import numpy as np
import pandas as pd
from math import exp


def initialize_network(data, n_out, n_hidden, n_neuron):
    len_row_data = len(data)
    wt_hl = [[random() for i in range(n_neuron + 1)] for i in range(n_hidden)]
    wt_ne = [[random() for i in range(n_neuron)] for i in range(n_out)]
    return wt_hl,wt_ne, n_out, n_hidden, n_neuron


def initialize_bias():
    print('')


def sumprod():
    print('')


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
initialize_network(data, n_out=len(data[len(data.columns) - 1].unique()), n_hidden=5, n_neuron=7)
print('asdfasd')
# below function will initilaize the initial weights for both the hidden neurons  and output layer neurons
