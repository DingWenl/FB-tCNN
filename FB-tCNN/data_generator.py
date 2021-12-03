from random import sample
from scipy.fftpack import fft,ifft
import random
import numpy as np
import keras

def datagenerator(batchsize,train_data_,win_train,y_lable_, start_time_, down_simple=4):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        for i in range(batchsize):
            sele = random.randint(0,(len(train_data_)-1))
            train_data = train_data_[sele]
            y_lable = y_lable_[sele]
            start_time = start_time_[sele]
            k = random.randint(0,(y_lable.shape[0]-1))
            y_data = y_lable[k]-1#标签
            time_start = random.randint(35,int(1000-win_train))
            x1 = int(start_time[k]/down_simple)+time_start
            x2 = int(start_time[k]/down_simple)+time_start+win_train
            x_1 = train_data[:,x1:x2]
            x_2 = np.reshape(x_1,(8, win_train, 1))
            x_train[i]=x_2         
            y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
        x_train = np.reshape(x_train,(batchsize,8, win_train, 1))
        y_train = np.reshape(y_train,(batchsize,4))
        
        yield x_train, y_train

def train_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,y_lable, start_time, down_simple,tran_list, channel):#, t_train):
    # gen_len = int(30 * t_train)
    while True:
        x_train1, x_train2, x_train3, x_train4, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
        for i in range(batchsize):
            k = sample(tran_list, 1)[0]
            # k = random.randint(0,(y_lable.shape[0]-1))
            y_data = y_lable[k]-1#标签
            time_start = random.randint(35,int(1000-win_train))
            # time_start = random.randint(35,int(1035-win_train))
            x1 = int(start_time[k]/down_simple)+time_start
            x2 = int(start_time[k]/down_simple)+time_start+win_train
            
            x_11 = train_data1[:,x1:x2]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            x_train1[i]=x_21
            
            x_12 = train_data2[:,x1:x2]
            x_22 = np.reshape(x_12,(channel, win_train, 1))
            x_train2[i]=x_22

            x_13 = train_data3[:,x1:x2]
            x_23 = np.reshape(x_13,(channel, win_train, 1))
            x_train3[i]=x_23

            # x_14 = train_data4[:,x1:x2]
            # x_24 = np.reshape(x_14,(channel, win_train, 1))
            # x_train4[i]=x_24

            y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
        x_train1 = np.reshape(x_train1,(batchsize,channel, win_train, 1))
        x_train2 = np.reshape(x_train2,(batchsize,channel, win_train, 1))
        x_train3 = np.reshape(x_train3,(batchsize,channel, win_train, 1))
        # x_train4 = np.reshape(x_train4,(batchsize,channel, win_train, 1))
        x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)
        y_train = np.reshape(y_train,(batchsize,4))
        
        # yield x_train1, x_train2, x_train3, x_train4, y_train
        yield x_train, y_train


def val_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,y_lable, start_time, down_simple,val_list, channel):#, t_train):
    # gen_len = int(30 * t_train)
    while True:
        x_train1, x_train2, x_train3, x_train4, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
        for i in range(batchsize):
            k = sample(val_list, 1)[0]
            # k = random.randint(0,(y_lable.shape[0]-1))
            y_data = y_lable[k]-1#标签
            time_start = random.randint(35,int(1000-win_train))
            # time_start = random.randint(35,int(1035-win_train))
            x1 = int(start_time[k]/down_simple)+time_start
            x2 = int(start_time[k]/down_simple)+time_start+win_train
            
            x_11 = train_data1[:,x1:x2]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            x_train1[i]=x_21
            
            x_12 = train_data2[:,x1:x2]
            x_22 = np.reshape(x_12,(channel, win_train, 1))
            x_train2[i]=x_22

            x_13 = train_data3[:,x1:x2]
            x_23 = np.reshape(x_13,(channel, win_train, 1))
            x_train3[i]=x_23

            # x_14 = train_data4[:,x1:x2]
            # x_24 = np.reshape(x_14,(channel, win_train, 1))
            # x_train4[i]=x_24

            y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
        x_train1 = np.reshape(x_train1,(batchsize,channel, win_train, 1))
        x_train2 = np.reshape(x_train2,(batchsize,channel, win_train, 1))
        x_train3 = np.reshape(x_train3,(batchsize,channel, win_train, 1))
        # x_train4 = np.reshape(x_train4,(batchsize,channel, win_train, 1))
        x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)
        y_train = np.reshape(y_train,(batchsize,4))
        
        # yield x_train1, x_train2, x_train3, x_train4, y_train
        yield x_train, y_train
# def sig_datagenerator(batchsize,train_data,win_train,y_lable, start_time, down_simple):#, t_train):
#     # gen_len = int(30 * t_train)
#     while True:
#         x_train, y_train = list(range(batchsize)), list(range(batchsize))
#         for i in range(batchsize):
        
#             k = random.randint(0,(y_lable.shape[0]-1))
#             y_data = y_lable[k]-1#标签
#             time_start = random.randint(35,int(35+1000-win_train))
#             x1 = int(start_time[k]/down_simple)+time_start
#             x2 = int(start_time[k]/down_simple)+time_start+win_train
#             x_1 = train_data[:,x1:x2]
#             x_2 = np.reshape(x_1,(8, win_train, 1))
#             x_train[i]=x_2         
#             y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
#         x_train = np.reshape(x_train,(batchsize,8, win_train, 1))
#         y_train = np.reshape(y_train,(batchsize,4))
        
#         yield x_train, y_train

#牛逼
# def sig_datagenerator(batchsize,train_data,win_train,y_lable, start_time, down_simple):#, t_train):
#     # gen_len = int(30 * t_train)
#     while True:
#         x_train, y_train = list(range(batchsize)), list(range(batchsize))
#         for i in range(batchsize):
        
#             k = random.randint(0,(y_lable.shape[0]-1))
#             y_data = y_lable[k]-1#标签
#             time_start = random.randint(35,int(35+1000-win_train))
#             x1 = int(start_time[k]/down_simple)+time_start
#             x2 = int(start_time[k]/down_simple)+time_start+win_train
#             x_1 = train_data[:,x1:x2]
#             # x_1 = np.array(x_1).T
#             x_2 = np.reshape(x_1,(win_train, 8, 1))
#             # x_2 = np.reshape(x_2,(1, 8, gen_len))
#             x_train[i]=x_2         
#             y_train[i] = keras.utils.to_categorical(y_data, num_classes=4, dtype='float32')
#         x_train = np.reshape(x_train,(batchsize,win_train, 8, 1))
#         # Gaussian_noise = np.random.normal(loc=0.0, scale=0.008, size=(batchsize,win_train, 8, 1))
#         # x_train = x_train + Gaussian_noise
#         # x_train = np.reshape(x_train,(batchsize,1, 8, gen_len))
#         # y_train = np.reshape(y_train,(batchsize, 1, 4))
#         y_train = np.reshape(y_train,(batchsize,4))
        
#         yield x_train, y_train
        # return x_train, y_train

