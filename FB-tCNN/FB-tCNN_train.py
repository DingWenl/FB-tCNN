from keras.callbacks import ModelCheckpoint
from net import fbtcnn_net
import data_generator
import scipy.io as scio 
from scipy import signal
from keras.models import Model
from keras.layers import Input
import numpy as np
from random import sample
import os

# get the filtered EEG-data, label and the start time of each trial of the dataset
def get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,wn14,wn24,path='D:/dwl/data/SSVEP_5.6/sess01_subj02_EEG_SSVEP.mat', down_sample=4):
    # read the data
    data = scio.loadmat(path)
    # get the EEG-data of the selected electrodes and downsampling it
    x_data = data['EEG_SSVEP_train']['x'][0][0][::down_sample,]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:,c]
    # get the label and onset time of each trial
    train_label = data['EEG_SSVEP_train']['y_dec'][0][0][0]
    train_start_time = data['EEG_SSVEP_train']['t'][0][0][0]
    # get the filtered EEG-data with six-order Butterworth filter of the first sub-filter
    channel_data_list1 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn11,wn21], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])  
        channel_data_list1.append(filtedData)
    channel_data_list1 = np.array(channel_data_list1)
    # get the filtered EEG-data with six-order Butterworth filter of the second sub-filter
    channel_data_list2 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn12,wn22], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])   
        channel_data_list2.append(filtedData)
    channel_data_list2 = np.array(channel_data_list2)
    # get the filtered EEG-data with six-order Butterworth filter of the third sub-filter
    channel_data_list3 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn13,wn23], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])   
        channel_data_list3.append(filtedData)
    channel_data_list3 = np.array(channel_data_list3)
    # get the filtered EEG-data with six-order Butterworth filter of the fourth sub-filter
    channel_data_list4 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn14,wn24], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:,i])  
        channel_data_list4.append(filtedData)
    channel_data_list4 = np.array(channel_data_list4)
    
    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, train_label, train_start_time

if __name__ == '__main__':
    # open the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    #%% Setting hyper-parameters
    # downsampling coefficient and sampling frequency after downsampling
    down_sample=4
    fs = 1000/down_sample
    # the number of the electrode channels
    channel = 9
    # the hyper-parameters of the training process
    train_epoch = 400
    batchsize = 256
    
    # the filter ranges of the four sub-filters in the filter bank
    f_down1 = 3
    f_up1 = 14
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 9
    f_up2 = 26
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 14
    f_up3 = 38
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs
    
    f_down4 = 19
    f_up4 = 50
    wn14 = 2*f_down4/fs
    wn24 = 2*f_up4/fs
    #%% Training the models of multi-subjects and multi-time-window
    # the list of the time-window
    t_train_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # selecting the training subject
    for sub_selelct in range(1, 10):
        # the path of the dataset and you need change it for your training
        path = 'D:/dwl/data/SSVEP_5.6/sess01/sess01_subj0%d_EEG_SSVEP.mat'%sub_selelct
        # get the filtered EEG-data of four sub-input, label and the start time of all trials of the training data
        data1, data2, data3, data4, label, start_time = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,wn14,wn24,path,down_sample)
        # selecting the training time-window
        for t_train in t_train_list:
            # transfer time to frame
            win_train = int(fs*t_train)
            # the traing data is randomly divided in the traning dataset and validation set according to the radio of 9:1
            train_list = list(range(100))
            val_list = sample(train_list, 10)
            train_list = [train_list[i] for i in range(len(train_list)) if (i not in val_list)]
            # data generator (generate the taining and validation samples of batchsize trials)
            train_gen = data_generator.train_datagenerator(batchsize,data1, data2, data3, data4,win_train,label, start_time, down_sample,train_list, channel)#, t_train)
            val_gen = data_generator.val_datagenerator(batchsize,data1, data2, data3, data4,win_train,label, start_time, down_sample,val_list, channel)#, t_train)
            #%% setting the input of the network
            input_shape = (channel, win_train, 4)
            input_tensor = Input(shape=input_shape)
            preds = fbtcnn_net(input_tensor)
            model = Model(input_tensor, preds)
            # the path of the saved model and you need to change it
            model_path = 'D:/dwl/github_code/FB-tCNN_and_tCNN_code/FB-tCNN/model/model_%3.1fs_0%d.h5'%(t_train, sub_selelct)
            # some hyper-parameters in the training process
            model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss',verbose=1, save_best_only=True,mode='auto')
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # training
            history = model.fit_generator(
                    train_gen,
                    steps_per_epoch= 10,
                    epochs=train_epoch,
                    validation_data=val_gen,
                    validation_steps=1,
                    callbacks=[model_checkpoint]
                    )

    #%% the rest of the training subjects, you maybe not need this part for your own dataset
    for sub_selelct in range(10, 55):
        
        path = 'D:/dwl/data/SSVEP_5.6/sess01/sess01_subj%d_EEG_SSVEP.mat'%sub_selelct
        data1, data2, data3, data4, label, start_time = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,wn14,wn24,path,down_sample)
        
        for t_train in t_train_list:
            win_train = int(fs*t_train)
            train_list = list(range(100))
            val_list = sample(train_list, 10)
            train_list = [train_list[i] for i in range(len(train_list)) if (i not in val_list)]
        
            train_gen = data_generator.train_datagenerator(batchsize,data1, data2, data3, data4,win_train,label, start_time, down_sample,train_list, channel)#, t_train)
            val_gen = data_generator.val_datagenerator(batchsize,data1, data2, data3, data4,win_train,label, start_time, down_sample,val_list, channel)#, t_train)
        #%%
            input_shape = (channel, win_train, 4)
            input_tensor = Input(shape=input_shape)
            preds = fbtcnn_net(input_tensor)
            model = Model(input_tensor, preds)
            
            model_path = 'D:/dwl/github_code/FB-tCNN_and_tCNN_code/FB-tCNN/model/model_%3.1fs_%d.h5'%(t_train, sub_selelct)
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





    # # show the process of the taining
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





