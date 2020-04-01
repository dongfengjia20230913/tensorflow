import tensorflow as tf
import numpy as np

print('----tf.constant----')
x = tf.constant([[2,2],[3,4]])#定义一个2x2的数组常量
print(x)#生成一个节点
print('num:',x.numpy())#输出常量数组值
print('shape:',x.shape)#输出常量数组尺寸
print('type:',x.dtype)#输出常量数组类型

print('----tf.ones----')
print(tf.ones(shape=(2,3)))#产生一个初始值皆为1的2x3的数组

print('----tf.zeros----')
print(tf.zeros(shape=(3,2)))#产生一个初始值皆为0的2x3的数组


#生成一个符合正态分布的2x2的数组，均值为0，标准差为1.0
print('----tf.random.normal----')

print(tf.random.normal(shape=(2,2),mean=0,stddev=1.0))


#生成一个数值在[0,10]的随机2x2的数组
print('----tf.random.uniform----')

print(tf.random.uniform(shape=(2,2),minval=0,maxval=10,dtype=tf.int32))
