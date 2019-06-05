#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:26:22 2018

@author: affine
"""
import os
# visualizing convnets
from keras.models import load_model
model = load_model("/home/affine/python/DL/models/cats_and_dogs_small_1.h5")
model.summary()

original_dataset_dir = '/home/affine/python/DL/CNN/kaggle_original_data/'
base_dir = '/home/affine/python/DL/CNN/cats_dogs_small'

test_dir = os.path.join(base_dir, 'test')
test_cats_dir = os.path.join(test_dir, 'cats')


img_path = test_cats_dir + "/" + "cat.1700.jpg"
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

from keras import models
# stacks all the layers one on top of another
layer_outputs = [layer.output for layer in model.layers[:8]]
# layer.output is the frame of Tensor. But why the "?" in (?, 150, 150, 3)
# combines all layers --- Keras distinguuishes the Input as non layer
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# activations
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]

import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,10], cmap='viridis')

### ------------ visualizing every channel in every intermediate activation
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
    images_per_row = 16
# layer_activation is the layer output: hence layer.output
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                        row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
            scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

### --------- visualizaing convnet filters

from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0
# getting to the layer output or 
layer_output = model.get_layer(layer_name).output
# shape of layer_output is shape=(?, ?, ?, 256)
loss = K.mean(layer_output[:,:,:,filter_index])
# loss is also a Tensor

# grads is also a function
grads = K.gradients(loss, model.input)[0]

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# we define a Keras back end function to get the loss and gradients which we
# calculate after passing the input matrix into the layer
iterate = K.function([model.input], [loss, grads]) # remember iterate is a func.
import numpy as np
# iterate is a function that takes a Numpy Tensor(as a list of tensors of size 1)
# and returns a list of 2 Numpy Tensors: the loss value and the gradient value
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# At this point we can define a Python Loop to do stochastic gradient descent

# starts from a gray image with some noise
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
step = 1 # magnitude of each gradient update

# Let's run gradient ascent for 40 steps
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data +=grads_value*step

# the resulting image tensor is a floating-point tensor of shape(1,150,150,3) 
# with values that may not be integers within [0,255]. 
# Hence you need to postprocess the image to turn it into displayable image
    
# Utility function to convert a tensor into a valid image
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# so the tensors are like mere frames where you can fit in input_image

def generate_pattern(layer_name, filter_index, size = 150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:, filter_index])
    
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([model.input], [loss, grads])
    
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
    step =1 
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value*step
        
    img = input_img_data[0]
    return deprocess_image(img)


# dismantling the function
layer_name = 'block3_conv1'
filter_index = 0
size = 150

layer_output = model.get_layer(layer_name).output

loss = K.mean(layer_output[:,:,:, filter_index])

#grads = K.gradients(loss, model.input)[0]
grads = K.gradients(loss, model.input)

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input], [loss, grads])

input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

step =1     

#i = 0

loss_value, grads_value = iterate([input_img_data])
input_img_data += grads_value*step

grads_value.shape

img = input_img_data[0]

plt.imshow(generate_pattern('block2_conv1', 0))

# responsive to polka dot pattern

layer_name = 'block1_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img
                
plt.figure(figsize=(20, 20))
plt.imshow(results)

# /home/affine/python/DL/CNN/cats_dogs_small/train/cats/cat.175.jpg
# /home/affine/python/DL/CNN/cats_dogs_small/train/dogs/dog.110.jpg

from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = "/home/affine/python/DL/CNN/cats_dogs_small/train/cats/cat.175.jpg"

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
np.argmin(preds[0])

african_elephant_output = model.output[:, 297]
last_conv_layer = model.get_layer('block5_conv3')

# find the gradient of the model output wrt the last layer output
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

import cv2

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

cv2.imwrite('/home/affine/python/DL/CNN/dogs_cam.jpg', superimposed_img)

