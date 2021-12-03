import keras
from keras.models import *
import scipy.io as scio 
import random
import numpy as np
from scipy import signal
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def datagenerator(batchsize,train_data,win_train,y_lable, start_time,down_simple, channel):
    x_train, y_train = list(range(batchsize)), list(range(batchsize))
    for i in range(batchsize):
    
        k = random.randint(0,(y_lable.shape[0]-1))
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
    return x_train, y_train



def get_test_data(wn1,wn2,path='D:/dwl/data/SSVEP_5.6/sess01_subj02_EEG_SSVEP.mat', down_simple=4):
    data = scio.loadmat(path)
    x_data = data['EEG_SSVEP_test']['x'][0][0][::down_simple,]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:,c]
    train_label = data['EEG_SSVEP_test']['y_dec'][0][0][0]
    train_start_time = data['EEG_SSVEP_test']['t'][0][0][0]
    
    channel_data_list = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
        channel_data_list.append(filtedData)
    channel_data_list = np.array(channel_data_list)
    return channel_data_list, train_label, train_start_time

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    down_simple = 4
    fs = 1000/down_simple
    channel = 9
    train_epoch = 100
    batchsize = 1024
    f_down = 3
    f_up = 50
    wn1 = 2*f_down/fs
    wn2 = 2*f_up/fs
    
    total_av_acc_list = []
    # t_train_list = [1.0]
    t_train_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for sub_selelct in range(1, 10):
        
        path = 'D:/dwl/data/SSVEP_5.6/sess02/sess02_subj0%d_EEG_SSVEP.mat'%sub_selelct
        test_data, test_label, test_start_time = get_test_data(wn1, wn2, path, down_simple)
        av_acc_list = []
        for t_train in t_train_list:
            win_train = int(fs*t_train)
            model_path = 'D:/dwl/github_code/FB-tCNN_and_tCNN_code/tCNN/model/model_0.1_%3.1fs_-0%d.h5'%(t_train, sub_selelct)
            model = load_model(model_path)
            print("load successed")
            print(t_train, sub_selelct)
            acc_list = []
            for j in range(5):
                x_train,y_train = datagenerator(batchsize, test_data, win_train, test_label, test_start_time, down_simple, channel)#, t_train)
                a, b = 0, 0
                y_pred = model.predict(np.array(x_train))
                true, pred = [], []
                y_true = y_train
                for i in range (batchsize-1):
                    y_pred_ = np.argmax(y_pred[i])
                    pred.append(y_pred_)
                    y_true1  = np.argmax(y_train[i])
                    true.append(y_true1)
                    if y_true1 == y_pred_:
                        a += 1
                    else:
                        b += 1
                acc = a/(a+b)
                acc_list.append(acc)
            av_acc = np.mean(acc_list)
            print(av_acc)
            av_acc_list.append(av_acc)
        total_av_acc_list.append(av_acc_list)

    for sub_selelct in range(10, 55):
        
        path = 'D:/dwl/data/SSVEP_5.6/sess02/sess02_subj%d_EEG_SSVEP.mat'%sub_selelct
        test_data, test_label, test_start_time = get_test_data(wn1, wn2, path, down_simple)
        av_acc_list = []
        for t_train in t_train_list:
            win_train = int(fs*t_train)
            model_path = 'D:/dwl/github_code/FB-tCNN_and_tCNN_code/tCNN/model/model_0.1_%3.1fs_-%d.h5'%(t_train, sub_selelct)
            model = load_model(model_path)
            print("load successed")
            print(t_train, sub_selelct)
            acc_list = []
            for j in range(5):
                x_train,y_train = datagenerator(batchsize, test_data, win_train, test_label, test_start_time, down_simple, channel)#, t_train)
                a, b = 0, 0
                y_pred = model.predict(np.array(x_train))
                true, pred = [], []
                y_true = y_train
                # print(y_train[2])
                for i in range (batchsize-1):
                    y_pred_ = np.argmax(y_pred[i])
                    pred.append(y_pred_)
                    y_true1  = np.argmax(y_train[i])
                    true.append(y_true1)
                    if y_true1 == y_pred_:
                        a += 1
                    else:
                        b+= 1
                acc = a/(a+b)
                acc_list.append(acc)
            av_acc = np.mean(acc_list)
            print(av_acc)
            av_acc_list.append(av_acc)
        total_av_acc_list.append(av_acc_list)
        
        
    # # save the results
    # print(total_av_acc_list)
    # company_name_list = total_av_acc_list
    # df = pd.DataFrame(company_name_list)
    # df.to_excel("D:/dwl/results_0.1/public/tcnn/sess01_to_02/tcnn_9channel_sess01_to_02.xlsx", index=False)