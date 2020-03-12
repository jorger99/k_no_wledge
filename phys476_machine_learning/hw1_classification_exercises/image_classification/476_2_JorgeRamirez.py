#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


"""
This jupyter notebook is based on using machine learning 
to classify written images of digits
"""


# In[2]:


""" 
this cell is for importing the data from keras.datasets

2 tuples:
    x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
    y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).

renamed datasets for clarity
    x_name -> name_in (input data)
    y_name -> name_out (output data)

"""

  # load data from keras.datasets
(train_in_unshaped, train_out_unencoded), (test_in_unshaped, test_out_unencoded) = mnist.load_data()

  # reshape all 60,000 28x28 vectors into 1x784 vectors
train_in = train_in_unshaped.reshape(60000, 784)
test_in = test_in_unshaped.reshape(10000, 784)


  # manually one hot encode numbers 0-9
shape = (train_out_unencoded.size, train_out_unencoded.max()+1)
train_out = np.zeros(shape) 
train_out[np.arange(train_out_unencoded.size), train_out_unencoded] = 1 

  # repeat for testing data
shape = (test_out_unencoded.size, test_out_unencoded.max()+1)
test_out = np.zeros(shape)
test_out[np.arange(test_out_unencoded.size), test_out_unencoded] = 1 


"""
plot the numbers :)
print(train_in[2])
plt.imshow(train_in[2], interpolation='nearest')
#plt.imshow(train_in[np.random.randint(0, train_in[1].size)], interpolation='nearest')
"""


# In[3]:


""" 
making a net with several hidden layers and some dropout sprinkled in because that's cool
"""

  # initialize a linear network structure
model = Sequential()

  # add some layers
model.add(Dense(units=1024, activation='relu', input_dim=(784)))

model.add(Dense(units=512, activation='relu'))

model.add(Dense(units=256, activation='relu'))

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=10, activation='softmax'))


  # configure learning process
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

  # train model for a few epochs. x is the input, y is the target (TRAINING) data
history = model.fit(x=train_in, y=train_out, epochs=5, verbose=2, batch_size=16)


# In[4]:


# evaluate performance, x is input and y is target (TESTING) data
    
loss_and_metrics_1 = model.evaluate(x=test_in, y=test_out, verbose=1)

print("\n\nResult of testing on test data:")
print(model.metrics_names)
print(loss_and_metrics_1)
print("\n")


# Plot training & validation loss values
plt.plot(history.history['accuracy'], label="Accuracy")

plt.title('History of Net Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

