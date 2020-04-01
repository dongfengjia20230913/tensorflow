# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

#y = w*x + b，我们对模型y = 3x+8进行预测
#产生随机样本数据


points= np.genfromtxt("lable.csv", delimiter=",")

N = len(points)

print(N)

for i in range(0,N):
    print(i, points[i,0], points[i,1])