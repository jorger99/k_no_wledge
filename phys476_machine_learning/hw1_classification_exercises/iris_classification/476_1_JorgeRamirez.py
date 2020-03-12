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

"""
This jupyter notebook is based on using machine learning 
to classify species of flowers from a canonical data set
"""


# In[ ]:


""" 
this cell is for importing the data, assumed to be in the same directory as .py file

   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
"""

  # load .data file into a pandas dataframe
filepath = "bezdekIris.data"
data = pd.read_csv(filepath, header=None, names=['sepal length','sepal width', 'petal length',
                                                        'petal width','class'])

  # one hot encode the classifications
data = pd.concat([pd.get_dummies(data['class'], prefix='class'), data], axis=1)  
del data['class']  # remove column with strings


# In[ ]:


"""
This cell is for creating training and testing datasets
"""

  # first take a random 90% of the data, use it as our training data
train_data = data.sample(frac = 9/10, axis = 0)

  # take everything else to be our testing data
test_data = data.drop(train_data.index)
  
    
"""
train_in will be the input data we use to train our network
train_out will be used as the answer key for the training done on train_in
"""
  # save columns that have the answer into train_out
train_out = train_data[['class_Iris-setosa', 'class_Iris-versicolor', 'class_Iris-virginica']].values  

  # now delete the columns we just saved and save the rest as input
del train_data['class_Iris-versicolor']
del train_data['class_Iris-virginica']
del train_data['class_Iris-setosa']
train_in = train_data.values


""" 
test_in will be the input parameters for seeing if our network works
test_out will be the answer key for test_in
"""
  # save columns that have the answer into test_out
test_out = test_data[['class_Iris-setosa', 'class_Iris-versicolor', 'class_Iris-virginica']].values

  # now delete the columns we just saved into train_out and save the rest as input
del test_data['class_Iris-versicolor']
del test_data['class_Iris-virginica']
del test_data['class_Iris-setosa']
test_in = test_data.values


  # clean up and delete unused arrays
del train_data
del test_data


# In[ ]:


""" 
making a net with several hidden layers and some dropout sprinkled in because that's cool
"""

  # initialize a linear network structure
model = Sequential()

  # add some layers
model.add(Dense(units=64, activation='relu', input_dim=4))

model.add(Dense(units=32, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(units=16, activation='relu'))

model.add(Dense(units=8, activation='relu'))

model.add(Dense(units=3, activation='softmax'))


  # configure learning process
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

  # train model for a few epochs. x is the input, y is the target (TRAINING) data
history = model.fit(x=train_in, y=train_out, epochs=15, verbose=2, batch_size=16)


# In[ ]:


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

