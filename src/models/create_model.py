#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from keras import layers
from keras import models
from keras.utils import np_utils


# In[2]:


def create_model(first_layer_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(512, kernel_size=5, strides=1,
                            padding="same", activation="relu",
                            input_shape=(first_layer_shape, 1)))
    
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

    model.add(layers.Conv1D(512, kernel_size=5, strides=1,
                            padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

    model.add(layers.Conv1D(256, kernel_size=5, strides=1,
                            padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

    model.add(layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides = 2, padding = 'same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(7, activation="softmax"))

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    
    return model


# In[ ]:




