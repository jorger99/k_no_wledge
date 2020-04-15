#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import History
from keras.utils import plot_model, to_categorical
from keras.backend import clear_session

from keras import optimizers

import matplotlib.pyplot as plt

''' global plotting settings '''
#plt.style.use('seaborn-paper')
# Update the matplotlib configuration parameters:
plt.rcParams.update({'text.usetex': False,
                     'lines.linewidth': 3,
                     'font.family': 'sans-serif',
                     'font.serif': 'Helvetica',
                     'font.size': 14,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'axes.labelsize': 'large',
                     'axes.titlesize': 'large',
                     'axes.grid': True,
                     'grid.alpha': 0.53,
                     'lines.markersize': 12,
                     'legend.borderpad': 0.2,
                     'legend.fancybox': True,
                     'legend.fontsize': 'medium',
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.1,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'figure.figsize': (12,8),
                     'savefig.dpi': 100,
                     'pdf.compression': 9})

"""
This jupyter notebook is based on using machine learning 
to categorize pictures of flowers (hw2 prob 2)


Jorge Ramirez
"""


# In[2]:


""" 
this cell is for importing the data, assumed to be in the same directory as .py file
"""

  # load .data file into a np array
    
flower_imgs = np.load("flower_imgs.npy")
flower_labels = np.load("flower_labels.npy")


"""
show the flowers
flower_no = 2  #index in array
print(flower_labels[flower_no])
plt.imshow(flower_imgs[flower_no], interpolation='nearest')
"""


# In[3]:


"""
This cell is for creating training and testing datasets, 
the train/test sets will be separated using keras in the model.compile()
"""

raw_data = flower_imgs  # (32, 32, 3) vectors (RGB images)
input_data = raw_data
output_data = flower_labels

# data is already sorted by channel_last 


# In[4]:


"""
need to shuffle input and output arrays but keep them consistent between each other
"""

  # thanks stackoverflow
def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


shuffle_in_unison_scary(input_data, output_data)


# In[5]:


""" 
making a net with several hidden layers
"""

  # initialize a linear network structure
model = Sequential()

  # add some layers
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(32, 32, 3), activation="sigmoid"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation="sigmoid"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=3, input_shape=(32, 32, 3), activation="sigmoid"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=5, activation="softmax"))


  # set a custom optimizer
#thats_our_boy = optimizers.RMSprop(learning_rate=0.0008)
thats_our_boy = optimizers.Nadam(learning_rate=0.001)
#thats_our_boy = keras.optimizers.Adam(learning_rate=0.001

  # configure learning process
model.compile(loss='sparse_categorical_crossentropy', optimizer=thats_our_boy, metrics=['accuracy'])

  # train model for a few epochs. x is the input, y is the target (TRAINING) data
history = model.fit(x=input_data, y=output_data, epochs=50, verbose=2, batch_size=32, validation_split=0.1)


# In[6]:


# evaluate performance, x is input and y is target (TESTING) data

# Plot training & validation loss values
plt.plot(history.history['accuracy'], label="Training")
plt.plot(history.history['val_accuracy'], label="Validation")

#plt.yscale("log")
plt.title('History of Net Accuracy')
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('HistoryofLoss')
plt.show()


# In[7]:

model.summary()

