from random import sample
import random
import numpy as np
import keras

# get the training sampels
def train_datagenerator(batchsize,train_data1,train_data2,train_data3,train_data4,win_train,y_label, start_time, down_sample,tran_list, channel):
    while True:
        x_train1, x_train2, x_train3, x_train4, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
        # get training samples of batchsize trials
        for i in range(batchsize):
            # randomly selecting the single-trial
            k = sample(tran_list, 1)[0]
            y_data = y_label[k]-1
            # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time, 1000 frames is data range we used
            time_start = random.randint(35,int(1000+35-win_train))
            x1 = int(start_time[k]/down_sample)+time_start
            x2 = int(start_time[k]/down_sample)+time_start+win_train
            # get four sub-inputs
            x_11 = train_data1[:,x1:x2]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            x_train1[i]=x_21
            
            x_12 = train_data2[:,x1:x2]
            x_22 = np.reshape(x_12,(channel, win_train, 1))
            x_train2[i]=x_22

            x_13 = train_data3[:,x1:x2]
            x_23 = np.reshape(x_13,(channel, win_train, 1))
            x_train3[i]=x_23

            x_14 = train_data4[:,x1:x2]
            x_24 = np.reshape(x_14,(channel, win_train, 1))
            x_train4[i]=x_24

            y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
            
        x_train1 = np.reshape(x_train1,(batchsize,channel, win_train, 1))
        x_train2 = np.reshape(x_train2,(batchsize,channel, win_train, 1))
        x_train3 = np.reshape(x_train3,(batchsize,channel, win_train, 1))
        x_train4 = np.reshape(x_train4,(batchsize,channel, win_train, 1))
        # concatenate the four sub-input into one input to make it can be as the input of the FB-tCNN's network
        x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4), axis=-1)
        y_train = np.reshape(y_train,(batchsize,4))
        
        yield x_train, y_train

# get the validation samples
def val_datagenerator(batchsize,train_data1,train_data2,train_data3,train_data4,win_train,y_label, start_time, down_sample,val_list, channel):
    while True:
        x_train1, x_train2, x_train3, x_train4, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
        for i in range(batchsize):
            k = sample(val_list, 1)[0]
            y_data = y_label[k]-1
            time_start = random.randint(35,int(1000-win_train))
            x1 = int(start_time[k]/down_sample)+time_start
            x2 = int(start_time[k]/down_sample)+time_start+win_train
            
            x_11 = train_data1[:,x1:x2]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            x_train1[i]=x_21
            
            x_12 = train_data2[:,x1:x2]
            x_22 = np.reshape(x_12,(channel, win_train, 1))
            x_train2[i]=x_22

            x_13 = train_data3[:,x1:x2]
            x_23 = np.reshape(x_13,(channel, win_train, 1))
            x_train3[i]=x_23

            x_14 = train_data4[:,x1:x2]
            x_24 = np.reshape(x_14,(channel, win_train, 1))
            x_train4[i]=x_24

            y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
        x_train1 = np.reshape(x_train1,(batchsize,channel, win_train, 1))
        x_train2 = np.reshape(x_train2,(batchsize,channel, win_train, 1))
        x_train3 = np.reshape(x_train3,(batchsize,channel, win_train, 1))
        x_train4 = np.reshape(x_train4,(batchsize,channel, win_train, 1))
        x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4), axis=-1)
        y_train = np.reshape(y_train,(batchsize,4))
        
        yield x_train, y_train


