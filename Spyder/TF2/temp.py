# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:53:26 2020

@author: Administrator
"""


import numpy as np

#激活函数
def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

print(sigmoid(21))