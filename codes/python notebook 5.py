#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:05:27 2018

@author: affine
"""

import os, shutil

original_dataset_dir = '/home/affine/python/DL/CNN/kaggle_original_data/'
base_dir = '/home/affine/python/DL/CNN/cats_dogs_small'

os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
# /home/affine/python/DL/CNN/kaggle_original_data/train/cat.0.jpg
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

# create separate directories for Train, Test and Validation
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)


test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
''' CATS '''

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

# copies 1st 1000 images of cats for train
for fname in fnames:
    src = os.path.join(original_dataset_dir + "/train", fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# copies next 1000 to 1500 images of cats for validation
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + "/train", fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# copies next 1000 to 1500 images of cats for test    
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + "/train", fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
''' DOGS '''

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]

# copies 1st 1000 images of cats for train
for fname in fnames:
    src = os.path.join(original_dataset_dir + "/train", fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copies next 1000 to 1500 images of cats for validation
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + "/train", fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copies next 1000 to 1500 images of cats for test    
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + "/train", fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# check if we have the right number of images for each of the folders
print('total number of Cat iamges', len(os.listdir(validation_dogs_dir)))

from keras import layers
from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

from keras import optimizers
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics= ['acc'])

### ---------- Data Pre-processing
# (1) Read the Picture Files
# (2) Decode the jpeg content to RGB grid of pixels
# (3) Convert these into floating point tensors
# (4) Rescale the pixel values which are between 0 and 255 to [0, 1] - as know that
#    networks prefer to work on small values
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./ 255)
test_datagen = ImageDataGenerator(rescale = 1./ 255)


# rather than converting all of the images to tensors we are doing it in batches
# generators are helping us convert them in batches
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150,150), # why do we resize to 150*150
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150,150),
        batch_size=20,
        class_mode='binary')



for data_batch, labels_batch in train_generator:
    print('data_batch_shape', data_batch.shape)
    print('labels_batch_shape', labels_batch.shape)
    break

# the generator and the infinite endlessness
def generator():
    i = 0
    while True:
        i += 1
        yield i

for item in generator():
    print(item)
    if item > 4:
        break

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

# it is a good practice to save your model after training
model.save('/home/affine/python/DL/models/cats_and_dogs_small_1.h5')

# display the curves of loss and accuracy during training
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# the plots are characteristic of overfitting. The Training Accuray increases
# linearly over time, until it reaches 100%, whereas Validation Accuracy stalls at
# ~.7. Validation Loss minimizes after 5 pochs
from keras import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3]
img = image.load_img(img_path, target_size = (150,150))
x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0

###############################################################################

########## --- adding Drop out ---


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'])

########## --- adding Drop out and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150,150),
        batch_size = 32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size= (150,150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

