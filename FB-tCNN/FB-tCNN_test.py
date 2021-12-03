from mne.io import read_raw_cnt
import mne
import pandas as pd
from scipy.fftpack import fft,ifft
import keras
from keras.models import *
import scipy.io as scio 
from sklearn.metrics import classification_report
import datagenerator_norest
import random
import numpy as np
import h5py
from scipy import signal
import os
from random import sample
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import time

def datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,y_lable, start_time, down_simple, test_list, channel):
    # gen_len = int(30 * t_trai
    x_train1, x_train2, x_train3, x_train4, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
    for i in range(batchsize):
        # k = sample(tran_list, 1)[0]
        k = sample(test_list, 1)[0]
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
    return x_train, y_train

# def get_test_data(wn11,wn21,wn12,wn22,wn13,wn23,wn14,wn24,input_fname, down_simple):
#     rawEEG = read_raw_cnt(input_fname, eog=(), misc=(), ecg=(), emg=(), data_format='auto', date_format='mm/dd/yy', preload=False, verbose=None)
#     tmp = rawEEG.to_data_frame()
#     tmp1 = tmp.values
#     tmp2 = tmp1[:,-8:]
#     train_data = tmp2[::down_simple,]
    
#     events_from_annot, _ = mne.events_from_annotations(rawEEG, event_id = 'auto')
#     train_label = [events_from_annot[i, 2] for i in range(events_from_annot.shape[0]) if (i%2 == 0)]
#     train_start_time = [events_from_annot[i, 0] for i in range(events_from_annot.shape[0]) if (i%2 == 1)]
    
#     channel_data_list1 = []
#     for i in range(train_data.shape[1]):
#         b, a = signal.butter(6, [wn11,wn21], 'bandpass')
#         filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
#         channel_data_list1.append(filtedData)
#     channel_data_list1 = np.array(channel_data_list1)
    
#     channel_data_list2 = []
#     for i in range(train_data.shape[1]):
#         b, a = signal.butter(6, [wn12,wn22], 'bandpass')
#         filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
#         channel_data_list2.append(filtedData)
#     channel_data_list2 = np.array(channel_data_list2)
    
#     channel_data_list3 = []
#     for i in range(train_data.shape[1]):
#         b, a = signal.butter(6, [wn13,wn23], 'bandpass')
#         filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
#         channel_data_list3.append(filtedData)
#     channel_data_list3 = np.array(channel_data_list3)
    
#     channel_data_list4 = []
#     for i in range(train_data.shape[1]):
#         b, a = signal.butter(6, [wn14,wn24], 'bandpass')
#         filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
#         channel_data_list4.append(filtedData)
#     channel_data_list4 = np.array(channel_data_list4)
    
#     return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, train_label, train_start_time

def get_test_data(wn11,wn21,wn12,wn22,wn13,wn23,input_fname, down_simple=4):
    rawEEG = read_raw_cnt(input_fname, eog=(), misc=(), ecg=(), emg=(), data_format='auto', date_format='mm/dd/yy', preload=False, verbose=None)
    tmp = rawEEG.to_data_frame()
    tmp1 = tmp.values
    tmp2 = tmp1[:,-8:]
    train_data = tmp2[::down_simple,]
    
    events_from_annot, _ = mne.events_from_annotations(rawEEG, event_id = 'auto')
    train_label = [events_from_annot[i, 2] for i in range(events_from_annot.shape[0]) if (i%2 == 0)]
    train_start_time = [events_from_annot[i, 0] for i in range(events_from_annot.shape[0]) if (i%2 == 1)]
    
    channel_data_list1 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn11,wn21], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
        channel_data_list1.append(filtedData)
    channel_data_list1 = np.array(channel_data_list1)
    
    channel_data_list2 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn12,wn22], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
        channel_data_list2.append(filtedData)
    channel_data_list2 = np.array(channel_data_list2)
    
    channel_data_list3 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn13,wn23], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
        channel_data_list3.append(filtedData)
    channel_data_list3 = np.array(channel_data_list3)
    
    
    return channel_data_list1, channel_data_list2, channel_data_list3, train_label, train_start_time

def get_test_list(block_n):
    if block_n == 0:
        test_list = list(range(20))
    if block_n == 1:
        test_list = list(range(20, 40))
    if block_n == 2:
        test_list = list(range(40, 60))
    if block_n == 3:
        test_list = list(range(60, 80))
    if block_n == 4:
        test_list = list(range(80, 100))
    if block_n == 5:
        test_list = list(range(100, 120))
    
    return test_list

if __name__ == '__main__':
    # pre_selelct = 2
    pc_ar = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    down_simple = 4
    fs = 1000/down_simple
    # t_train = 0.5
    # win_train = int(fs*t_train)
    channel = 8
    # train_epoch = 100
    batchsize = 1000
    f_down1 = 7
    f_up1 = 17
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 16
    f_up2 = 32
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 25
    f_up3 = 47
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs
    
    # t_train_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    total_av_acc_list = []
    t_train_list = [0.2]
    # t_train_list = [1.5]
    # t_train_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for pre_selelct in range(6,7):
        path = 'E:/better_data/0%d.cnt'%pre_selelct
        data1, data2, data3, label, start_time = get_test_data(wn11,wn21,wn12,wn22,wn13,wn23,path,down_simple)
        av_acc_list = []
        for t_train in t_train_list:
            win_train = int(fs*t_train)
            print(t_train, pre_selelct)
            
            acc_list = []
            for block_n in range(1):
                model_path = 'D:/dwl/code_ssvep/DL/my_data_code/FB_tcnn/model_0.1/better_model_%d_0%d_%3.1fs_%d.h5'%(pc_ar,pre_selelct, t_train, block_n)
                model = load_model(model_path)
                print("load successed")
            
                test_list = get_test_list(block_n)
                x_true, y_true = datagenerator(batchsize,data1, data2, data3,win_train,label, start_time, down_simple, test_list, channel)
                y_pred = model.predict(np.array(x_true))
                a, b = 0, 0
                pred, true = [], []
                for i in range (batchsize):
                    y_pred_ = np.argmax(y_pred[i])
                    pred.append(y_pred_)
                    y_true1  = np.argmax(y_true[i])
                    true.append(y_true1)
                    if y_true1 == y_pred_:
                        a += 1
                    else:
                        b+= 1
                acc = a/(a+b)
                print(acc)
                acc_list.append(acc)
            print(acc_list)
            mean_acc = np.mean(acc_list)
            print(mean_acc)
            av_acc_list.append(mean_acc)
        total_av_acc_list.append(av_acc_list)
        model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.])
    
# #%%保存数据excel
    print(total_av_acc_list)
    # company_name_list = total_av_acc_list
    # # company_name_list = [['腾讯', '北京'], ['阿里巴巴', '杭州'], ['字节跳动', '北京']]
    
    # 	 # list转dataframe
    # df = pd.DataFrame(company_name_list)
    #   # df = pd.DataFrame(company_name_list, columns=['company_name', 'local'])
    
    # 	 # 保存到本地excel
    # df.to_excel("D:/dwl/results_0.1/ours/fb_tcnn/fbtcnn_8channel_ours_.xlsx", index=False)

