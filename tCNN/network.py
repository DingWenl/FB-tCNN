# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:54:10 2020

@author: Administrator
"""
from keras import regularizers
from keras.layers import Conv2D,BatchNormalization,Dense,Activation,Dropout,Flatten

k = 4
L=regularizers.l2(0.01)


# tcnn
drop_out = 0.4
out_channel = 16
activation = 'elu'
def MLP_net(inputs):#shape=(None, 8, 250, 1)
    
    x = Conv2D(out_channel,kernel_size = (8, 1),strides = 1,padding = 'valid',kernel_regularizer = L)(inputs)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    
    x = Dropout(drop_out)(x)
    x = Conv2D(out_channel,kernel_size = (1, inputs.shape[2]),strides = 5,padding = 'same', kernel_regularizer = L)(x)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)

    x = Dropout(drop_out)(x)
    x = Conv2D(out_channel,kernel_size = (1, 5),strides = 1,padding = 'valid', kernel_regularizer = L)(x)#(None, 1, 25, 8) dtype=float32>
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)


    x = Dropout(drop_out)(x)
    x = Conv2D(32,kernel_size = (1, x.shape[2]),strides = 1,padding = 'valid', kernel_regularizer = L)(x)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    
    x = Flatten()(x)

    x = Dropout(drop_out)(x)
    x = Dense(k,activation='softmax')(x)#shape=(None, 1, 1, 4)

    return x







