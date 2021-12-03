from random import sample
import random
import numpy as np
import keras

def train_datagenerator(batchsize,train_data,win_train,y_lable, start_time, down_simple,train_list, channel):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        for i in range(batchsize):
            k = sample(train_list, 1)[0]
            y_data = y_lable[k]-1
            time_start = random.randint(35,int(1000-win_train))
            x1 = int(start_time[k]/down_simple)+time_start
            x2 = int(start_time[k]/down_simple)+time_start+win_train
            x_1 = train_data[:,x1:x2]
            x_2 = np.reshape(x_1,(channel, win_train, 1))
            x_train[i]=x_2         
            y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
        x_train = np.reshape(x_train,(batchsize,channel, win_train, 1))
        y_train = np.reshape(y_train,(batchsize,4))
        
        yield x_train, y_train


def val_datagenerator(batchsize,train_data,win_train,y_lable, start_time, down_simple,val_list, channel):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        for i in range(batchsize):
            k = sample(val_list, 1)[0]
            y_data = y_lable[k]-1
            time_start = random.randint(35,int(1000-win_train))
            x1 = int(start_time[k]/down_simple)+time_start
            x2 = int(start_time[k]/down_simple)+time_start+win_train
            x_1 = train_data[:,x1:x2]
            x_2 = np.reshape(x_1,(channel, win_train, 1))
            x_train[i]=x_2         
            y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
        x_train = np.reshape(x_train,(batchsize,channel, win_train, 1))
        y_train = np.reshape(y_train,(batchsize,4))
        
        yield x_train, y_train


