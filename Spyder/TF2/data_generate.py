# -*- coding: utf-8 -*-

"""
Created on Wed Apr  1 15:46:41 2020

@author: jiadongfeng-os
"""


import csv
import numpy as np
import pandas as pd 


#随机生成 y = 8x+3的数据，并保存到lebel文件中
x = np.random.normal(size=[100, 1], loc=3.0, scale=1.0)
#100条样子的目标值。人为设定权重8，偏置3.0
y = np.matmul(x, [[8.0]]) + 3.0


c = np.concatenate((x,y),axis=1)#axis=1表示对应行的数组进行拼接




with open('lable.csv','w',newline='') as f:
    write = csv.writer(f)
    write.writerows(c)