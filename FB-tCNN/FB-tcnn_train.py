from mne.io import read_raw_cnt
import mne
# import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from MLP import MLP_net
from keras.models import *
import random
import keras
# import json
import datagenerator_norest
# import scipy.io as scio 
from scipy import signal
# from keras import optimizers
from keras.models import Model
from keras.layers import Input
import numpy as np
from random import sample
import os
import matplotlib.pyplot as plt

def get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,input_fname, down_simple=4):
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

def get_train_list(block_n):
    totol_list = list(range(120))
    
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
    
    train_list = [totol_list[i] for i in range(len(totol_list)) if (totol_list[i] not in test_list)]
    
    return train_list



if __name__ == '__main__':
    # pre_selelct = 2
    pc_ar = 2
    down_simple = 4
    fs = 1000/down_simple
    # t_train = 0.5
    # gen_len = int(30 * t_train)
    # win_train = int(fs*t_train)
    channel = 8
    train_epoch = 200
    batchsize = 256
    
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
    
    # f_down4 = 19
    # f_up4 = 50
    # wn14 = 2*f_down4/fs
    # wn24 = 2*f_up4/fs
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    t_train_list = [0.2]
    # t_train_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # t_train_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    # t_train_list = [0.1,0.2, 0.3, 0.4, 0.5, 0.6]
    # t_train_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for pre_selelct in range(6, 7):
        path='E:/better_data/0%d.cnt'%pre_selelct
        data1, data2, data3, label, start_time = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path,down_simple)
            
        for t_train in t_train_list:
            win_train = int(fs*t_train)
            
            for block_n in range(1):
                train_list = get_train_list(block_n)
                val_list = sample(train_list, 10)
                train_list = [train_list[i] for i in range(len(train_list)) if (train_list[i] not in val_list)]
        
                train_gen = datagenerator_norest.train_datagenerator(batchsize,data1, data2, data3, win_train,label, start_time, down_simple,train_list, channel)#, t_train)
                val_gen = datagenerator_norest.val_datagenerator(batchsize,data1, data2, data3, win_train,label, start_time, down_simple,val_list, channel)#, t_train)
            #%%
                input_shape = (channel, win_train, 3)
                input_tensor = Input(shape=input_shape)
                preds = MLP_net(input_tensor)
                model = Model(input_tensor, preds)
                
                model_path = 'D:/dwl/code_ssvep/DL/my_data_code/FB_tcnn/model_0.1/better_model_%d_0%d_%3.1fs_%d.h5'%(pc_ar,pre_selelct, t_train, block_n)
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
            # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_2.5s_loss0%d.jpg'%pre_selelct)
            
            # plt.show()




