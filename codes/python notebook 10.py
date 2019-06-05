#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:52:37 2018

@author: affine
"""

from keras.models import load_model
model = load_model("/home/affine/python/DL/models/cats_and_dogs_small_1.h5")
model.summary()

layer_name = 'conv2d_46'
filter_index = 0
size = 150

layer_output = model.get_layer(layer_name).output

loss = K.mean(layer_output[:,:,:, filter_index])

#grads = K.gradients(loss, model.input)[0]
grads = K.gradients(loss, model.input)[0]

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input], [loss, grads])

input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

step =1     

#i = 0

loss_value, grads_value = iterate([input_img_data])
input_img_data += grads_value*step

grads_value.shape