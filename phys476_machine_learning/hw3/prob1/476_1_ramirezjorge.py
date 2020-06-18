#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn as nn

dtype = torch.float
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# In[ ]:


""" first prepare the datasets """

f = h5py.File("Galaxy10.h5", 'r')  # open h5py file

  # extract each dictionary entry into its own dataset
dset_img = f['images']
dset_ans = f['ans']      

''' uncomment to see raw data + plot of digit
print(dset_ans[2])
plt.imshow(dset_img[2], interpolation='nearest')
print(dset_img[2])
'''

  # convert to numpy first because it gets converted faster
numpy_img = np.array(dset_img)
numpy_ans = np.array(dset_ans)

  # also move the 3-channels to 1st position for pytorch in the img set
numpy_img = np.moveaxis(numpy_img, -1, 0)  # move channels to 2nd axis
numpy_img = np.moveaxis(numpy_img, 1, 0)  # move N datapoints to 1st axis
  
  # now convert these datasets to PyTorch tensors
data_in = torch.tensor(numpy_img, dtype=dtype, device=device)
data_out = torch.tensor(numpy_ans, dtype=dtype, device=device)

  # collect garbage
del dset_img, dset_ans, numpy_img, numpy_ans, f


# In[ ]:


print(torch.cuda.is_available())
print(data_in.shape)  # should be NxCxHxW


# In[ ]:


""" hyperparams """

batch_size = 8

learning_rate = 5e-6

epochs = 5


# In[ ]:


""" begin creation of neural network """

  # create our conv2d layers
first_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
second_conv = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
third_conv = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)

  # create our flattening & pooling layers
maxpool2d = nn.MaxPool2d(kernel_size=3)  # add one of these in between each conv2d layer


  # create model, not gonna lie it looks pretty ugly but hey man it's readable
model = nn.Sequential( 
    first_conv,
    nn.Sigmoid(),
    maxpool2d,
    
    second_conv,
    nn.Sigmoid(),
    maxpool2d,
    
    third_conv,
    nn.Sigmoid(),
    maxpool2d,
    
    nn.Flatten() 
    
).to(device)  # put on device

  # define our loss function
loss_fn = torch.nn.CrossEntropyLoss()

  # define our optimizer
optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


""" train neural network """
losses= []
for t in range(epochs):
    
      # run the nn once and save its predictions to y-pred
    y_pred = model(data_in)  
    
      # calculate our loss for this specific pass
    loss = loss_fn(y_pred, data_out.long())
    
      # print loss each epoch for debugging
    #if t % 10 == 9:
    #    print(t, loss.item())
    
      # zero the gradients 
    optimizer.zero_grad()
    
      # now backpropagate
    loss.backward()
    optimizer.step()
      
      # save individual epoch losses
    losses.append(loss.data.mean())
    
      # update all of the weights in each parameter Tensor that doesn't have no_grad()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            
    print('[%d] Loss: %.3f' % (t, np.mean(losses)))
    
    # not gonna lie, everything I try to look up about PyTorch is horribly complex or not relevant
    # I like keras so much more but that's probably a cop out

