from __future__ import print_function

from collections import defaultdict
from six.moves import range

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from os.path import join,exists

#np.random.seed(1337)

K.set_image_data_format('channels_first')

def build_generator(latent_size, seqlen, nchannel):
    model = Sequential()
    model.add(Dense(input_dim=latent_size, units=128))
    model.add(Activation('relu'))
    model.add(Dense(int(16*1*(seqlen/2))))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((16,1,int(seqlen/2)), input_shape=(int(16*1*(seqlen/2)),)))
    model.add(UpSampling2D(size=(1,2), data_format="channels_first"))
    model.add(Conv2D(nchannel, (1,3), padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    return model

def build_discriminator(seqlen, nchannel, output_activation = 'relu'):
    model = Sequential()
    model.add(Conv2D(16,(1,3), padding='same', input_shape=(nchannel,1,seqlen), data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation(output_activation))
    return model

