# import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from net import tcnn_net
from keras.models import *
# import random
# import keras
# import json
import data_generator
import scipy.io as scio 
from scipy import signal
# from keras import optimizers
from keras.models import Model
from keras.layers import Input
import numpy as np
from random import sample
import os

def get_train_data(wn1,wn2,path, down_simple=4):
    data = scio.loadmat(path)
    x_data = data['EEG_SSVEP_train']['x'][0][0][::down_simple,]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:,c]
    train_label = data['EEG_SSVEP_train']['y_dec'][0][0][0]
    train_start_time = data['EEG_SSVEP_train']['t'][0][0][0]
    
    channel_data_list = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])   #data为要过滤的信号
        channel_data_list.append(filtedData)
    channel_data_list = np.array(channel_data_list)
    return channel_data_list, train_label, train_start_time

def get_test_data(wn1,wn2,path, down_simple=4):
    data = scio.loadmat(path)
    x_data = data['EEG_SSVEP_test']['x'][0][0][::down_simple,]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:,c]
    train_label = data['EEG_SSVEP_test']['y_dec'][0][0][0]
    train_start_time = data['EEG_SSVEP_test']['t'][0][0][0]
    
    channel_data_list = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])
        channel_data_list.append(filtedData)
    channel_data_list = np.array(channel_data_list)
    return channel_data_list, train_label, train_start_time

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    down_simple=4
    fs = 1000/down_simple
    channel = 9
    train_epoch = 400
    batchsize = 256
    f_down = 3
    f_up = 50
    wn1 = 2*f_down/fs
    wn2 = 2*f_up/fs
    
    t_train_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for sub_selelct in range(1, 10):
        path='D:/dwl/data/SSVEP_5.6/sess01/sess01_subj0%d_EEG_SSVEP.mat'%sub_selelct
        data, label, start_time = get_train_data(wn1,wn2,path,down_simple)
        
        for t_train in t_train_list:
            win_train = int(fs*t_train)
            train_list = list(range(100))
            val_list = sample(train_list, 10)
            train_list = [train_list[i] for i in range(len(train_list)) if (i not in val_list)]
        
            train_gen = data_generator.train_datagenerator(batchsize,data,win_train,label, start_time, down_simple,train_list, channel)#, t_train)
            val_gen = data_generator.val_datagenerator(batchsize,data,win_train,label, start_time, down_simple,val_list, channel)#, t_train)
        #%%
            input_shape = (channel, win_train, 1)
            input_tensor = Input(shape=input_shape)
            preds = tcnn_net(input_tensor)
            model = Model(input_tensor, preds)
            
            model_path = 'D:/dwl/github_code/FB-tCNN_and_tCNN_code/tCNN/model/model_0.1_%3.1fs_-0%d.h5'%(t_train, sub_selelct)
            model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss',verbose=1, save_best_only=True,mode='auto')
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            history = model.fit_generator(
                    train_gen,
                    steps_per_epoch= 10,
                    epochs=train_epoch,
                    validation_data=val_gen,
                    validation_steps=1,
                    callbacks=[model_checkpoint]
                    )
            print(t_train,sub_selelct)

    for sub_selelct in range(10, 55):
        path='D:/dwl/data/SSVEP_5.6/sess01/sess01_subj%d_EEG_SSVEP.mat'%sub_selelct
        data, label, start_time = get_train_data(wn1,wn2,path,down_simple)
        
        for t_train in t_train_list:
            win_train = int(fs*t_train)
            train_list = list(range(100))
            val_list = sample(train_list, 10)
            train_list = [train_list[i] for i in range(len(train_list)) if (train_list[i] not in val_list)]
        
            train_gen = data_generator.train_datagenerator(batchsize,data,win_train,label, start_time, down_simple,train_list, channel)#, t_train)
            val_gen = data_generator.val_datagenerator(batchsize,data,win_train,label, start_time, down_simple,val_list, channel)#, t_train)
        #%%
            input_shape = (channel, win_train, 1)
            input_tensor = Input(shape=input_shape)
            preds = tcnn_net(input_tensor)
            model = Model(input_tensor, preds)
            
            model_path = 'D:/dwl/github_code/FB-tCNN_and_tCNN_code/tCNN/model/model_0.1_%3.1fs_-%d.h5'%(t_train, sub_selelct)
            model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss',verbose=1, save_best_only=True,mode='auto')
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            history = model.fit_generator(
                    train_gen,
                    steps_per_epoch= 10,
                    epochs=train_epoch,
                    validation_data=val_gen,
                    validation_steps=1,
                    callbacks=[model_checkpoint]
                    )
            print(t_train,sub_selelct)
    
    
    
    
    
    
    # # show the process of the training
    # epochs=range(len(history.history['loss']))
    # plt.subplot(221)
    # plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
    # plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
    # plt.title('Traing and Validation accuracy')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_V3.1_acc1.jpg')
    
    # plt.subplot(222)
    # plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    # plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_2.5s_loss0%d.jpg'%sub_selelct)
    
    # plt.show()





