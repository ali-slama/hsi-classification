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


def  load_dataset(dataset):
    """load dataset parameters from config.json"""
    
    with open('./config.json') as f:
        config = json.loads(f.read())
        params = config[dataset]
        data = sio.loadmat(params['img_path'])[params['img']]
        labels = sio.loadmat(params['gt_path'])[params['gt']]
        num_classes = params['num_classes']
        target_names = params['target_names']
        
    return data,labels,num_classes,target_names

def apply_pca(X, num_components=75):
    """apply pca to X and return new_X"""
    
    new_X = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_X = pca.fit_transform(new_X)
    new_X = np.reshape(new_X, (X.shape[0],X.shape[1], num_components))
    return new_X, pca

def pad_with_zeros(X, margin=2):
    """apply zero padding to X with margin"""
    
    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X

def create_patches(X, y, window_size=5, remove_zero_labels = True):
    """create patch from image. suppose the image has the shape (w,h,c) then the patch shape is
    (w*h,window_size,window_size,c)"""
    
    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    # split patches
    patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchs_labels = np.zeros((X.shape[0] * X.shape[1]))
    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patches_data[patch_index, :, :, :] = patch
            patchs_labels[patch_index] = y[r-margin, c-margin]
            patch_index = patch_index + 1
    if remove_zero_labels:
        patches_data = patches_data[patchs_labels>0,:,:,:]
        patchs_labels = patchs_labels[patchs_labels>0]
        patchs_labels -= 1
    return patches_data, patchs_labels

def split_train_test_set(X, y, test_ratio=0.10):
    """split dataset into train set and test set with test_ratio"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

#Parameters
dataset = "Indian_pines" # Indian_pines or PaviaU or or Salinas  . check config.json
window_size = 25
num_pca_components = 30
test_ratio = 0.7

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

model.save('./saved_models/model_hybrid_{}_{}_{}_{}.h5'.format(dataset,window_size,num_pca_components,test_ratio))