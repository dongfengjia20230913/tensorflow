import tensorflow as tf
import numpy as np

print('----tf.constant----')

#定义一个2x2的正态数组变量
variable = tf.random.normal(shape=(2,2))

a = tf.Variable(variable)#创建一个变量，默认值为variable


print('----show varialble----')
print(a)

new_value = tf.random.normal(shape=(2, 2))

a.assign(new_value)#赋值，使得两个变量的值相同


for i in range(2):
    for j in range(2):
         assert a[i, j] == new_value[i, j]
         
         