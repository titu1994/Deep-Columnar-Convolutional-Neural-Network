from keras.layers import merge, Input, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D

img_rows, img_cols = 32, 32

nb_filters_1 = 96
nb_filters_2 = 192
nb_filters_3 = 256
nb_conv = 1
nb_conv_mid = 3
nb_conv_init = 5

def conv(init, nb_filter, row, col, subsample=(1,1), repeat=0):
    c = Convolution2D(nb_filter, row, col, border_mode='same', subsample=subsample)(init)
    c = LeakyReLU()(c)

    for i in range(repeat):
        c = Convolution2D(nb_filter, row, col, border_mode='same', subsample=subsample)(c)
        c = LeakyReLU()(c)
    return c


init = Input(shape=(3, img_rows, img_cols),)

fork11 = conv(init, nb_filters_1, nb_conv_init, nb_conv_init, repeat=1)
fork12 = conv(init, nb_filters_1, nb_conv_mid, nb_conv_mid, repeat=1)
merge1 = merge([fork11, fork12, ], mode='concat', concat_axis=1)
conv_pool1 = conv(merge1, nb_filters_1, nb_conv_init, nb_conv_init, subsample=(2,2))
bn1 = BatchNormalization(axis=1)(conv_pool1)

fork21 = conv(bn1, nb_filters_2, nb_conv_mid, nb_conv_mid, repeat=1)
fork22 = conv(bn1, nb_filters_2, nb_conv_mid, nb_conv_mid, repeat=1)
merge2 = merge([fork21, fork22, ], mode='concat', concat_axis=1)
conv_pool2 = conv(merge2, nb_filters_2, nb_conv_mid, nb_conv_mid, subsample=(2,2))
bn2 = BatchNormalization(axis=1)(conv_pool2)

fork31 = conv(bn2, nb_filters_3, nb_conv_mid, nb_conv_mid)
fork32 = conv(bn2, nb_filters_3, nb_conv, nb_conv)
fork33 = conv(bn2, nb_filters_3, nb_conv_init, nb_conv_init)
merge3 = merge([fork31, fork32, fork33, ], mode='concat', concat_axis=1)
conv_pool3 = conv(merge3, nb_filters_2, nb_conv_mid, nb_conv_mid, subsample=(2,2))
dropout = Dropout(0.5)(conv_pool3)

flatten = Flatten()(dropout)
output = Dense(10, activation="softmax")(flatten)

model = Model(input=init, output=output)

model.summary()