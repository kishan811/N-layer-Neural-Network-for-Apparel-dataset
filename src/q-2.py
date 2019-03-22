#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import random
from sklearn.model_selection import train_test_split


# In[12]:


df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
df.head()
# df.shape


# Neural Networks can be very much applied to regression problem.<br>
# In case of regression problem, use of softmax activation or any kind of activation is not required at the last layer.

# ### For predicting the selling price of the house:-
# #### We will use neural network with 3 layers.
# #### We will Use ‘ReLU’ as the activation function for the hidden layers because ReLU is linear function and its  value is max(0,x).
# #### We can use 'Mean squared Error' as a loss function. (Instead of Cross-entropy Error.)
# #### We will define the output layer with only one node to predict the House selling price. (Unlike 10 output layers in Apparel dataset.)
# #### We will Use Linear activation function like ReLU for the output layer as we want ouput as a linear prediction value based on the model.

# #### <u>Structure:</u>
#     -- Remove Nan Values from data set and Columns that contain Nan Values.
#     -- For categorical attributes, One hot Encode data. i.e. for n categories in column make n rows with in exactly 1 category. 
#     1)1 input layer with input units equal to the number of features in dataset used to predict house price + 1 (for the bias unit).<br>
#     2)1 hidden layer - Keeping it simple. Assuming we don’t have complex data.<br>
#     3)Output layer with 1 unit giving us the prediction sale price.<br>
#     4)Use Quadratic loss function to predict how well the neural network is performing. <br>
#         L(w)=∑i(h(xi,w)−yi)^2
#     5)Use Gradient Descent to minimize error and change weights.

# In[ ]:




