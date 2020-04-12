# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:47:40 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from label_image_load import load_dataset

print ("***Load and reshape data:***")

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# size of train set
m_train = train_set_y.shape[1]
#size of test set
m_test = test_set_y.shape[1]
#image px size
num_px = train_set_x_orig.shape[1]

print ("show image shape:")

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# Reshape the training and test examples
# START CODE HERE  (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# END CODE HERE 

print ("reduce image dimen:")

print ("train set shape: " + str(train_set_x_flatten.shape))
print ("train set label shape: " + str(train_set_y.shape))
print ("test set shape: " + str(test_set_x_flatten.shape))
print ("test set label shape: " + str(test_set_y.shape))

print ("data set uniform:")


train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

print ("train_set_x shpae: " + str(train_set_x.shape))
print ("test_set_x shpae: " + str(test_set_x.shape))


print ("***Define sigmod and lass function:***")

#激活函数
def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

#激活函数求导
def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)


#损失函数
def loss_function(y_pred, y_true):
  data_size = y_true.shape[1]
  return  (- 1 / data_size) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))  # compute cost


#print(loss_function(train_set_y,train_set_y))
print ("***Define NeuralNetwork:***")

class OurNeuralNetwork:
 
  def __init__(self,dim):
    # Weights
    np.random.seed(1000)
    self.w = np.random.rand(dim,1)
    self.b = 0
    assert(self.w.shape == (dim, 1))
    assert(isinstance(self.b, float) or isinstance(self.b, int))
    
    
  def initialize_with_zeros(self, dim):
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros(shape=(dim, 1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
 
    return w, b
       
  def feedforward(self,  w, b, x):
    sum_o1 = np.dot(w.T, x) + b  # compute activation
    o1 = sigmoid(sum_o1)  # compute activation
    return o1


  def propagate(self,w, b, X, y_true,y_pred):

    image_num = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    print("all cose:",np.sum(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred))))
    cost = (- 1 / image_num) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))  # compute cost
    ### END CODE HERE ###
    
 
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    print("dddw",np.dot(X, (y_pred - y_true).T))
    dw = (1 / image_num) * np.dot(X, (y_pred - y_true).T)
    db = (1 / image_num) * np.sum(y_pred - y_true)
    
    print("dw",dw)

    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


  def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        
        y_pred = nework.feedforward(w, b, X)  # compute activation

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = self.propagate(w, b, X, Y, y_pred)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


  def predict(self, w, b, X):
    
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


  def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
   
    
    ### START CODE HERE ###
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = self.initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = self.predict(w, b, X_test)
    Y_prediction_train = self.predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


nework = OurNeuralNetwork(10)

# feedforward and propagate test case
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])

y_pred = nework.feedforward(w, b, X)  # compute activation

print ("y_pred = " , y_pred)

grads, cost = nework.propagate(w, b, X, Y, y_pred)

print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


#params, grads, costs = nework.optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

#print ("w = " + str(params["w"]))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))


#d = nework.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)