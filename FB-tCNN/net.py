# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:54:10 2020

@author: Administrator
"""
from keras import regularizers
import keras
from keras.layers import Conv2D,BatchNormalization,Dense,Activation,Dropout,Flatten,GlobalMaxPooling2D,Lambda#,regularizers#GlobalMaxPooling2D


k = 4
L=regularizers.l2(0.01)
channel =9
lenth = 5
drop_out = 0.4
out_channel = 16

def slice_(x,index):
        return x[:,:,:,index]

activation = 'elu'
def fbtcnn_net(inputs):#shape=(None, 8, 250, 4)
    inputs1 = Lambda(slice_,arguments={'index':0})(inputs)
    inputs1 = keras.layers.core.Reshape((inputs1.shape[1],inputs1.shape[2],1))(inputs1)
    inputs2 = Lambda(slice_,arguments={'index':1})(inputs)
    inputs2 = keras.layers.core.Reshape((inputs2.shape[1],inputs2.shape[2],1))(inputs2)
    inputs3 = Lambda(slice_,arguments={'index':2})(inputs)
    inputs3 = keras.layers.core.Reshape((inputs3.shape[1],inputs3.shape[2],1))(inputs3)
    inputs4 = Lambda(slice_,arguments={'index':3})(inputs)
    inputs4 = keras.layers.core.Reshape((inputs4.shape[1],inputs4.shape[2],1))(inputs4)
#%% share the weights
    conv_1 = Conv2D(out_channel,kernel_size = (channel, 1),strides = 1,padding = 'valid',kernel_regularizer = L)
    conv_2 = Conv2D(out_channel,kernel_size = (1, inputs.shape[2]),strides = 5,padding = 'same', kernel_regularizer = L)
    conv_3 = Conv2D(out_channel,kernel_size = (1, lenth),strides = 1,padding = 'valid', kernel_regularizer = L)
#%%
    x1 = conv_1(inputs1)
    x1 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x1)
    x1 = Activation(activation)(x1)
    
    x1 = Dropout(drop_out)(x1)
    x1 = conv_2(x1)
    x1 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x1)
    x1 = Activation(activation)(x1)

    x1 = Dropout(drop_out)(x1)
    x1 = conv_3(x1)
    x1 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x1)
    x1 = Activation(activation)(x1)
#%%
    x2 = conv_1(inputs2)
    x2 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x2)
    x2 = Activation(activation)(x2)
    
    x2 = Dropout(drop_out)(x2)
    x2 = conv_2(x2)
    x2 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x2)
    x2 = Activation(activation)(x2)

    x2 = Dropout(drop_out)(x2)
    x2 = conv_3(x2)
    x2 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x2)
    x2 = Activation(activation)(x2)
#%%
    x3 = conv_1(inputs3)
    x3 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x3)
    x3 = Activation(activation)(x3)
    
    x3 = Dropout(drop_out)(x3)
    x3 = conv_2(x3)
    x3 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x3)
    x3 = Activation(activation)(x3)

    x3 = Dropout(drop_out)(x3)
    x3 = conv_3(x3)
    x3 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x3)
    x3 = Activation(activation)(x3)
#%%
    x4 = conv_1(inputs4)
    x4 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x4)
    x4 = Activation(activation)(x4)
    
    x4 = Dropout(drop_out)(x4)
    x4 = conv_2(x4)
    x4 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x4)
    x4 = Activation(activation)(x4)

    x4 = Dropout(drop_out)(x4)
    x4 = conv_3(x4)
    x4 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x4)
    x4 = Activation(activation)(x4)
#%%相加
    x = keras.layers.add([x1, x2, x3, x4])
#动态卷积
    x = Dropout(drop_out)(x)
    x = Conv2D(32,kernel_size = (1, x.shape[2]),strides = 1,padding = 'valid', kernel_regularizer = L)(x)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    #降低维度
    x = Flatten()(x)
    #全连接层加softmax
    x = Dropout(drop_out)(x)
    x = Dense(k,activation='softmax')(x)#shape=(None, 1, 1, 4)
    return x
