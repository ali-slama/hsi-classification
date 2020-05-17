import os
import random
import itertools
from random import shuffle
import h5py
import json

import numpy as np
import scipy
import scipy.io as sio # Scipy input and output
import scipy.ndimage 
from skimage.transform import rotate 
import spectral # Module for processing hyperspectral image data.
import matplotlib 


# scikit-learn imports 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# keras imports 
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from Utils.common import load_dataset, apply_pca, create_patches, split_train_test_set



os.chdir('../')


def train_and_save_model():
    X, y , num_classes , target_names = load_dataset(dataset)
    print("Initial {}".format(X.shape))

    X,pca = apply_pca(X,num_pca_components)
    print("PCA {}".format(X.shape))

    X_patches, y_patches = create_patches(X, y, window_size=window_size)
    print("Patches {} ".format(X_patches.shape))

    X_train, X_test, y_train, y_test = split_train_test_set(X_patches, y_patches, test_ratio)
    print("Split {}".format(X_train.shape))


    y_train = np_utils.to_categorical(y_train) # convert class labels to on-hot encoding
    y_test = np_utils.to_categorical(y_test)

    X_train = X_train.reshape(-1,window_size,window_size,num_pca_components,1)
    X_train.shape


    ## input layer
    input_layer = Input((window_size, window_size, num_pca_components,1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32,kernel_size=(3,3,3),activation='relu')(conv_layer2)

    conv3d_shape = conv_layer3._keras_shape
    conv_layer3 = Reshape((conv3d_shape[1],conv3d_shape[2],conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    dense_layer1 = Dense(units=256,activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128,activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)

    output_layer = Dense(units=num_classes,activation='softmax')(dense_layer2)
        
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    adam = Adam(lr=0.001,decay=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=256, epochs=50)

    model.save('../saved_models/model_hybrid_{}_{}_{}_{}.h5'.format(dataset,window_size,num_pca_components,test_ratio))


with open('./scripts/parameters.json') as f:
    file_content = json.loads(f.read())
    parameters = file_content['parameters']

for params in parameters :
    dataset = params['dataset']
    window_size = params['window_size']
    num_pca_components = params['num_pca_components']
    test_ratio = params['test_ratio']
    train_and_save_model()
