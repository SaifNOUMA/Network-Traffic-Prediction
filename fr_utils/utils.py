
#%%
import math
import numpy as np
from glob import glob
import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.svm import SVR

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, GRU
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, mean_squared_logarithmic_error
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import mean_absolute_percentage_error as MAPE


#%%

def read_data (path = 'data/fdata_Timestep_60'):
    with open(path) as f:
        lines = f.read().splitlines()
    series = np.array(list(map(int, lines)))
    return series

def train_test_split(dataset, train_frac):
    """
    :param dataset: A series of Network Traffic Volume
    :param train_frac: The percentage of the training set from the whole data
    :return: Couple of (Train set, Test set)
    """
    train_size = int(len(dataset)*train_frac)
    return dataset[:train_size], dataset[train_size:]



def create_datasets(dataset, look_back=1, look_ahead=1):
    """
    :param dataset: The series of Network Traffic Volume (Nb of Packets / Length of Packets)
    :param look_back: The window size which represents the number of steps that the model will use them as input
    :return: Couples consists of ( look_back Values of the series, The observed value of the series )
    """
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-look_ahead+1):
        window = dataset[i:(i+look_back)]
        data_x.append(window)
        data_y.append(dataset[i + look_back:i + look_back + look_ahead])
    return np.array(data_x), np.array(data_y)

def plot_series(time, series, format="-", start=0, end=None, figsize=(10,6), xlabel="Time", ylabel="Paclets per Second", path="test.png"):
    """
    :param time: represents the granularity of time ( 1 second or 1 Minute)
    :param series: The values of the network traffic volume over time
    :param format: The shape of the line
    :param start: The lower bound of the time interval
    :param end: The upper bound of the time interval
    :param figsize: Size of the figure
    :param xlabel: Name of x-axis
    :param ylabel: Name of y-axis
    :return:  Plot of the time series which represents in our case the network traffic volume
    """
    fig = plt.figure(1, figsize)
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    # plt.savefig(path)
    # plt.close()

def plot_hist(data, bins = 20 , xlabel = "" , ylabel = "" , title = "", path="test1.png"):
    """
    :param data: The values of the network traffic volume over time
    :param bins: The sequence, or the edges presented in the figure
   :param xlabel: Name of x-axis
    :param ylabel: Name of y-axis
    :param title:
    :return: A histogram represent the distribution of the data
    """
    plt.hist(data, bins, color = 'green',
            edgecolor = 'black')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()
    plt.savefig(path)
    plt.close()

def build_seq2seq_model(look_ahead = 1):
    m = Sequential()
    # encoder
    m.add(GRU(16, input_dim = 1))
    # repeat for the number of steps out
    m.add(RepeatVector(look_ahead))
    # decoder
    m.add(GRU(8, return_sequences=True))
    m.add(GRU(8, return_sequences=True))
    # split the output into timesteps
    m.add(TimeDistributed(Dense(1)))
    m.compile(loss='mse', optimizer='rmsprop')
    #m.summary()
    return m

def reverse_scale(data, mean, std):
    for x in np.nditer(data, op_flags=['readwrite']):
        x[...] = x*std + mean
    return data

def calculate_error(train_y, test_y, pred_train, pred_test):
    test_score = math.sqrt(mean_squared_error(test_y, pred_test))
    train_score = math.sqrt(mean_squared_error(train_y, pred_train))
    return train_score, test_score

def mean_absolute_percentage(y, y_pred):
    return np.mean(np.abs((y - y_pred) / y)) * 100


def plot_1_error(pred_test, test_y, er1, path="test.png"):

    fig = plt.figure(1, (18, 13))
    test_y  = test_y.reshape(len(test_y))
    pred_test = pred_test.reshape(len(pred_test))
    plt.plot(test_y, label="Observed")
    plt.plot(pred_test, color="red", label="Predicted, MAPE: " + str(round(er1, 5)) + "%")
    plt.title("1 step ahead prediction")
    plt.ylabel("Number of Packets / minute")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)
    plt.savefig(path)
    plt.close()

def plot_4_errors(pred_test, test_y, er1, er2, er3, er4, path="test.png"):
    fig = plt.figure(1, (18, 13))
    plt.subplot(221)
    plt.plot(test_y[:, 0, :], label="Observed")
    plt.plot(pred_test[:, 0, :], color="red", label="Predicted, MAPE: " + str(round(er1, 5)) + "%")
    plt.title("1 step ahead prediction")
    plt.ylabel("Number of Packets / minute")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.subplot(222)
    plt.plot(pred_test[:, 3, :], color="red", label="Predicted, MAPE: " + str(round(er2, 5)) + "%")
    plt.plot(test_y[:, 3, :], label="Observed")
    plt.title("4 step ahead prediction")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.subplot(223)
    plt.plot(pred_test[:, 7, :], color="red", label="Predicted, MAPE: " + str(round(er3, 5)) + "%")
    plt.plot(test_y[:, 7, :], label="Observed")
    plt.title("8 step ahead prediction")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    plt.subplot(224)
    plt.plot(pred_test[:, 15, :], color="red", label="Predicted, MAPE: " + str(round(er4, 5)) + "%")
    plt.plot(test_y[:, 15, :], label="Observed")
    plt.title("16 step ahead prediction")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)

    # plt.show()
    plt.savefig(path)
    plt.close()

