from keras.layers import merge, Input
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D

img_rows, img_cols = 32, 32

nb_filters_1 = 64
nb_filters_2 = 128
nb_filters_3 = 256
nb_conv = 3
nb_conv_mid = 4
nb_conv_init = 5

init = Input(shape=(3, img_rows, img_cols))

fork11 = Convolution2D(nb_filters_1, nb_conv_init, nb_conv_init, activation="relu", border_mode='same')(init)
fork12 = Convolution2D(nb_filters_1, nb_conv_init, nb_conv_init, activation="relu", border_mode='same')(init)
merge1 = merge([fork11, fork12, ], mode='concat', concat_axis=1)
conv_pool1 = Convolution2D(nb_filters_1, nb_conv_init, nb_conv_init, activation="relu", subsample=(2,2), border_mode='same')(merge1)

fork21 = Convolution2D(nb_filters_2, nb_conv_mid, nb_conv_mid, activation="relu", border_mode='same')(conv_pool1)
fork22 = Convolution2D(nb_filters_2, nb_conv_mid, nb_conv_mid, activation="relu", border_mode='same', )(conv_pool1)
merge2 = merge([fork21, fork22, ], mode='concat', concat_axis=1)
conv_pool2 = Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", subsample=(2,2), border_mode='same')(merge2)

fork31 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(conv_pool2)
fork32 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(conv_pool2)
fork33 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(conv_pool2)
fork34 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(conv_pool2)
fork35 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(conv_pool2)
fork36 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(conv_pool2)
merge3 = merge([fork31, fork32, fork33, fork34, fork35, fork36, ], mode='concat', concat_axis=1)
conv_pool3 = Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", subsample=(2,2), border_mode='same', name='conv_pool3')(merge3)

dropout = Dropout(0.5)(conv_pool3)

flatten = Flatten()(dropout)
output = Dense(10, activation="softmax")(flatten)

model = Model(input=init, output=output)

model.summary()
