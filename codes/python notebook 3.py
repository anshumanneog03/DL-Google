#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:59:37 2018

@author: affine
"""
#network = models.Sequential()
#network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
#network.add(layers.Dense(10, activation='softmax'))
#
#network.compile(optimizer='rmsprop',
#        loss='categorical_crossentropy',
#        metrics=['accuracy'])

# Cross Validation and Epochs
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

train_data.shape
test_data.shape

# Normalizing the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model

# ---------------------------------------------------------------------------
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28,1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)

test_acc
# the dense layer
# the number of input features = 2, x1 and x2
# single layer NN with 3 nodes
#Dense Layer meaning: relu(dot(W,x)+b)
# W = [[w1[1], w1[2], w1[3]], [w1[1], w2[2]]],
# X = [[x(1)1,x(1)2], [x(2)1,x(2)2],[x(3)1,x(3)2],[x(4)1,x(4)2],[x(5)1,x(5)2]]
# dot(W, X) is the representation of data in 3 dimentional hyperplane
# which gives: [Node 1
# [[w1[1]x(1)1+w2[1]x(1)2],
# [w1[1]x(2)1+w2[1]x(2)2],
# [w1[1]x(3)1+w2[1]x(3)2],
# [w1[1]x(4)1+w2[1]x(4)2],
# [w1[1]x(5)1+w2[1]x(5)2]]

# for Node 2:
# [[w1[2]x(1)1+w2[2]x(1)2],
# [w1[2]x(2)1+w2[2]x(2)2],
# [w1[2]x(3)1+w2[2]x(3)2],
# [w1[2]x(4)1+w2[2]x(4)2],
# [w1[2]x(5)1+w2[2]x(5)2]]

# for Node 3:
# [[w1[3]x(1)1+w2[3]x(1)2],
# [w1[3]x(2)1+w2[3]x(2)2],
# [w1[3]x(3)1+w2[3]x(3)2],
# [w1[3]x(4)1+w2[3]x(4)2],
# [w1[3]x(5)1+w2[3]x(5)2]]

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D)