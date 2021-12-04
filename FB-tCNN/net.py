from keras import regularizers
import keras
from keras.layers import Conv2D,BatchNormalization,Dense,Activation,Dropout,Flatten,GlobalMaxPooling2D,Lambda#,regularizers#GlobalMaxPooling2D

# Setting hyper-parameters
k = 4
L=regularizers.l2(0.01)
channel =9
lenth = 5
drop_out = 0.4
out_channel = 16

# Slice the input concatenated in data_generator to restore it to four sub-inputs
def slice_(x,index):
        return x[:,:,:,index]

activation = 'elu'
def fbtcnn_net(inputs):
    # Slice the input concatenated in data_generator to restore it to four sub-inputs
    inputs1 = Lambda(slice_,arguments={'index':0})(inputs)
    inputs1 = keras.layers.core.Reshape((inputs1.shape[1],inputs1.shape[2],1))(inputs1)
    inputs2 = Lambda(slice_,arguments={'index':1})(inputs)
    inputs2 = keras.layers.core.Reshape((inputs2.shape[1],inputs2.shape[2],1))(inputs2)
    inputs3 = Lambda(slice_,arguments={'index':2})(inputs)
    inputs3 = keras.layers.core.Reshape((inputs3.shape[1],inputs3.shape[2],1))(inputs3)
    inputs4 = Lambda(slice_,arguments={'index':3})(inputs)
    inputs4 = keras.layers.core.Reshape((inputs4.shape[1],inputs4.shape[2],1))(inputs4)
    #%% share the weights of the sub-inputs' three convolution layers
    conv_1 = Conv2D(out_channel,kernel_size = (channel, 1),strides = 1,padding = 'valid',kernel_regularizer = L)
    conv_2 = Conv2D(out_channel,kernel_size = (1, inputs.shape[2]),strides = 5,padding = 'same', kernel_regularizer = L)
    conv_3 = Conv2D(out_channel,kernel_size = (1, lenth),strides = 1,padding = 'valid', kernel_regularizer = L)
    #%% one sub-branch
    # the first shared convolution layer
    x1 = conv_1(inputs1)
    x1 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x1)
    x1 = Activation(activation)(x1)
    # the second shared convolution layer
    x1 = Dropout(drop_out)(x1)
    x1 = conv_2(x1)
    x1 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x1)
    x1 = Activation(activation)(x1)
    # the third shared convolution layer
    x1 = Dropout(drop_out)(x1)
    x1 = conv_3(x1)
    x1 = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x1)
    x1 = Activation(activation)(x1)
    #%% one sub-branch
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
    #%% one sub-branch
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
    #%% one sub-branch
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
    #%% the four sub-features are fused (added)
    x = keras.layers.add([x1, x2, x3, x4])
    # the fourth convolution layer
    x = Dropout(drop_out)(x)
    x = Conv2D(32,kernel_size = (1, x.shape[2]),strides = 1,padding = 'valid', kernel_regularizer = L)(x)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    # flentten used to reduce the dimension of the features
    x = Flatten()(x)
    # the fully connected layer and "softmax"
    x = Dropout(drop_out)(x)
    x = Dense(k,activation='softmax')(x)#shape=(None, 1, 1, 4)
    
    return x
