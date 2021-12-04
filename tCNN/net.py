from keras import regularizers
from keras.layers import Conv2D,BatchNormalization,Dense,Activation,Dropout,Flatten

# Setting hyper-parameters
k = 4
L=regularizers.l2(0.01)
drop_out = 0.4
out_channel = 16
activation = 'elu'

# the network of the tCNN
def tcnn_net(inputs):
    # the first convolution layer
    x = Conv2D(out_channel,kernel_size = (9, 1),strides = 1,padding = 'valid',kernel_regularizer = L)(inputs)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    # the second convolution layer
    x = Dropout(drop_out)(x)
    x = Conv2D(out_channel,kernel_size = (1, inputs.shape[2]),strides = 5,padding = 'same', kernel_regularizer = L)(x)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    # the third convolution layer
    x = Dropout(drop_out)(x)
    x = Conv2D(out_channel,kernel_size = (1, 5),strides = 1,padding = 'valid', kernel_regularizer = L)(x)#(None, 1, 25, 8) dtype=float32>
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    # the fourth convolution layer
    x = Dropout(drop_out)(x)
    x = Conv2D(32,kernel_size = (1, x.shape[2]),strides = 1,padding = 'valid', kernel_regularizer = L)(x)
    x = BatchNormalization(axis = -1,momentum = 0.99,epsilon=0.001)(x)
    x = Activation(activation)(x)
    # flentten used to reduce the dimension of the features
    x = Flatten()(x)

    x = Dropout(drop_out)(x)
    # the fully connected layer and "softmax"
    x = Dense(k,activation='softmax')(x)

    return x
