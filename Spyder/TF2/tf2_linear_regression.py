import numpy as np
import matplotlib.pyplot as plt


#获取样本数据
points= np.genfromtxt("lable.csv", delimiter=",")
N = len(points)



sl = 0.01;#学习率


w=0
b=0
epoch_number = 1000

for epoch in range(epoch_number):
    deriv_w, deriv_b = 0., 0.
    # 计算所有样本的梯度平均值
    for i in range(0, N):
        x = points[i,0]
        y_true = points[i,1]
        deriv_w += 2/N * (w * x +b - y_true)* x
        deriv_b += 2/N * (w * x +b - y_true)*1
    # 利用所有样本梯度的平均值更新w,b
    w = w - sl  * deriv_w
    b = b - sl  * deriv_b
    # 每隔100步 计算一下当前的损失值
    if epoch % 100 == 0:
        current_loss = 0.
        for i in range(0, N):
            y_pred = w * points[i, 0] + b 
            y_true = points[i,1]
            current_loss = current_loss + 1/N * (y_pred  - y_true)**2
        print('epoch :',epoch, 'current_loss:', current_loss,'w = ',w,'b = ',b)
# 对所有样本迭代完100次后 输出最后的w,b
        
        
#画出真是的值，和回归后的曲线
    
x_true = []
y_true = []
y_pred = []    
for i in range(0, N):
        x_true.append(points[i,0])
        y_true.append(points[i,1])
        y_pred.append(w*points[i,0]+b)
        
print(x_true)
plt.plot(x_true,y_true,"ro")
plt.plot(x_true,y_pred)

plt.show()

