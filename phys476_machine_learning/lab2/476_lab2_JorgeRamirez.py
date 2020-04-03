#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.callbacks import History
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt


from keras.datasets import mnist
from matplotlib.ticker import MaxNLocator


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
to reconstruct vectors after they have been shrunk
"""


# In[ ]:


""" 
this cell is for generating datasets

"""

  # generate some random vectors, 10k that are 1x20
input_size = 10000
    
  # first take a random 90% of the data, use it as our training data
train_data = np.random.rand(int(input_size), 20)

train_in, train_out = train_data, train_data  # identical in/out


  # take everything else to be our testing data
test_data = np.random.rand(int(input_size), 20)

test_in, test_out = test_data, test_data


# In[ ]:


""" 
making a net with several hidden layers and some dropout sprinkled in because that's cool
"""

  # initialize a linear network structure
model = Sequential()

  # add some layers
model.add(Dense(units=20, activation='relu', input_dim=(20)))

model.add(Dense(units=18, activation='relu'))

model.add(Dense(units=14, activation='relu'))

model.add(Dense(units=10, activation='relu'))

model.add(Dense(units=8, activation='relu'))

model.add(Dense(units=10, activation='relu'))

model.add(Dense(units=14, activation='relu'))

model.add(Dense(units=18, activation='relu'))

model.add(Dense(units=20, activation='sigmoid'))

  # configure learning process
model.compile(loss='mse', optimizer='Adam')

  # train model for a few epochs. x is the input, y is the target (TRAINING) data
history = model.fit(x=train_in, y=train_out, epochs=50, verbose=2, batch_size=1, validation_data=[test_in, test_out])


# In[ ]:


# evaluate performance, x is input and y is target (TESTING) data
    
print("\n\nResult of testing on test data:")
print(model.metrics_names)
print("\n")


# Plot training & validation loss values
plt.plot(history.history['loss'], label="training")
plt.plot(history.history['val_loss'], label="testing")

plt.axhline(0.05, color="r", label="Target")

plt.title('History of Net Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

