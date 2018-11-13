# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.1
#   kernelspec:
#     display_name: Python (dl)
#     language: python
#     name: dl
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.6
# ---

# ## Flowers image classification workshop

# ### Basic task in computer vision

# #### Image classification

# #### Image localization

# #### Image segmentation

# ### Image classification

# #### How do human recognize car?
# - Rectangular-box shape
# - 4 wheels
# - Pair of headlights
# - Pair of Tail lights
# - etc...
#
# #### How do human differenciate between car make/model?
# - Hard to tell...
# - Rond vs carré
# - Numbre de porte
# - Type de grillage

# #### Concolutional Neural Network (CNN)
# ##### Conceptually
# Instagram filter
# Edge Detection
# ##### Learning the kernel - CNN
# Conv
# ##### Inception v3
# Network

# ### Flowers dataset
# 4 242 images of flowers. Data is based on Flicr, Google Images and Yandex Image.
# Images are split into 5 categories
# - Chamomile
# - Tulip
# - Rose
# - Sunflower
# - Dandelion
#
# Every classes has about 800 images. Dimension of image isn't fixed.

# +
# %load_ext autoreload
# %autoreload 2\

import os
import random

from skimage.io import imread, imshow

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

# -

def show_images_horizontal(path, n=5):
    files = random.sample(os.listdir(path), 5)
    images = list()
    
    for file in files:
        images.append(mpimg.imread(path + file))
    
    plt.figure(figsize=(20, 10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

# #### Daisy

path = "data/raw/daisy/"
show_images_horizontal(path)

path = "data/raw/dandelion/"
show_images_horizontal(path)

path = "data/raw/roses/"
show_images_horizontal(path)

path = "data/raw/sunflowers/"
show_images_horizontal(path)

path = "data/raw/tulips/"
show_images_horizontal(path)

os.listdir("data/raw")

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy

train_datagen = image.ImageDataGenerator()
val_datagen = image.ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
    directory=r"data/processed/train",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=966
)

val_generator = val_datagen.flow_from_directory(
    directory=r"data/processed/val",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=966
)


inception = InceptionV3(weights='imagenet', include_top=False)

x = inception.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
out = Dense(5, activation='softmax')(x)

model = Model(inputs=inception.input, outputs=out)

for layer in inception.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["categorical_accuracy"])

model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.n//train_generator.batch_size,
                    validation_data=val_generator,
                    validation_steps=val_generator.n//val_generator.batch_size,
                    epochs=1,
                    verbose=1
)

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["categorical_accuracy"])

model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.n//train_generator.batch_size,
                    validation_data=val_generator,
                    validation_steps=val_generator.n//val_generator.batch_size,
                    epochs=10,
                    verbose=1
)
