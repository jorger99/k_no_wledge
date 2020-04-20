#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import torch

from keras.datasets import mnist

"""
This jupyter notebook is based on using PyTorch to
classify 28x28 grayscale image data into an integer from 0-9
"""

var = 0   # so jupyterlab doesnt output the string above


# In[ ]:


"""
this cell is for importing the data from keras.datasets
and then casting them to tensors for use with PyTorch
"""

  # load data from keras.datasets
(train_in_unshaped, train_out_unencoded), (test_in_unshaped, test_out_unencoded) = mnist.load_data()

''' uncomment to see raw data + plot of digit
print(train_out_unencoded[2])
print(train_in_unshaped[2])
plt.imshow(train_in_unshaped[2], interpolation='nearest')
'''

  # reshape all 60,000 28x28 vectors into 1x784 vectors
np_train_in = train_in_unshaped.reshape(60000, 784)
np_test_in = test_in_unshaped.reshape(10000, 784)

  # cast our boiz to make sure no errors occur

  # convert to torch tensors
train_in = torch.from_numpy(np_train_in).float() / 255
train_out = torch.from_numpy(train_out_unencoded).long() / 255

test_in = torch.from_numpy(np_test_in).float() / 255
test_out = torch.from_numpy(test_out_unencoded).long() / 255


# In[ ]:


""" network hyperparameters """

batch_size = 64  # batch size
Dim_in = 784  # input dimension of network: row vector consisting of flattened images
Dim_hid = 400 # dimension of hidden layer
Dim_out = 10  # output dimension of network: row vector consisting of one-hot-encoded numbers 0-9

epochs = 100

learning_rate = 1e-5

""" pytorch parameters """

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")  # run on GPU? :O


# In[ ]:


""" construct the net """
'''
input_layer     -> 784 x 1   (each image is 784x1)
hidden_layer_1  -> 400 x 1    
output_layer    -> 10 x 1    (expecting one hot encoded output)
'''

nnet = torch.nn.Sequential(
    torch.nn.Linear(Dim_in, Dim_hid),
    torch.nn.ReLU(),
    torch.nn.Linear(Dim_hid, Dim_out),
    )

def model(xb):
    return xb @ weights + bias
    
def loss_fn(ypred, out):
    CE = torch.nn.functional.cross_entropy
    loss = CE(ypred, out)
    return loss

def accuracy(ypred, out):
    preds = torch.argmax(ypred, dim=1)
    accs = (preds == out).float()
    
    return accs.mean()


  # first randomly initialize weights, use autograd to backpropagate
weight_layer1 = torch.randn(size=(Dim_in, Dim_hid), device=device, dtype=dtype, requires_grad=True) / np.sqrt(Dim_in)
weight_layer2 = torch.randn(size=(Dim_hid, Dim_out), device=device, dtype=dtype, requires_grad=True) / np.sqrt(Dim_hid)

""" train neural net for certain number of epochs """


for step in range(epochs):
    # first, taking the training_in data and matrix multiply (.mm()) by the first set of weights
    # next, our activation function takes the form of .clamp forcing all our negatives to be zero
    # finally, matrix multiply by the second set of weights
    y_pred = nnet(train_in)
    
    # now compute our loss function by taking the square sum of the difference between the answer and our guess
    cat_loss = loss_fn(y_pred, train_out)
    acc = accuracy(y_pred, train_out)
    
    if step % 10 == 0:
        print("Epoch: {}, Loss: {:.2f}, Accuracy: {:.3f}".format(step, cat_loss.item(), acc.item()))
    
    # run autograd backpropagation
    cat_loss.backward()
    
    # manually update weights with gradient descent
    
    with torch.no_grad():
        for param in nnet.parameters():
            param -= learning_rate * param.grad
    

