import numpy as np

def one_hot_fun(N):
    a = np.random.randint(0, 99, N)
    b = np.zeros((a.size, 100)) #创建N * 100 的零矩阵
    b[np.arange(a.size), a] = 1 #每一行的对应列数值为一
    return b

def anti_fun(l):
    b = np.where(l == 1) #得到数值为1的对应行、列坐标
    return b[1] #返回列
N =int(input("请输入样本数N："));
b = one_hot_fun(N)
print(b)
print(anti_fun(b))