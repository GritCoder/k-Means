import numpy as np
import matplotlib.pyplot as plt
"""
梯度下降demo 以凸函数y=x^2为例来简单实现一下其过程
"""
def func(x):#目标函数y=x^2
    return np.square(x)
def dfunc(x):#目标函数的导数
    return 2 * x
def GD(x_start,epochs,lr,df=dfunc):
    """
       梯度下降法。给定起始点与目标函数的一阶导函数，求在epochs次迭代中x的更新值
       :param x_start: x的起始点
       :param df: 目标函数的一阶导函数
       :param epochs: 迭代周期(次数)
       :param lr: 学习率
       :return: x在每次迭代后的位置(取值)（包括起始点），长度为epochs+1
    """
    xs = np.zeros(epochs + 1)#返回的列表
    x = x_start#起始迭代点
    xs[0] = x#返回列表的第一个元素为x的起始值
    for i in range(epochs):#开始迭代
        dx = df(x)#其实严格来说，这里应该输入原函数，但是这里仅作为demo演示过程，已经提前固定好原函数和导数了 读者知道即可
        v = - dx * lr#每次迭代要改变的值
        x += v#累加改变值
        xs[i+1] = x #把更新后的取值存进列表中
    return xs #最后返回列表
def demo_GD():#演示如何使用梯度下降法
    line_x = np.linspace(-5,5,100)
    line_y = func(line_x)
    x_start = -5
    epochs = 5
    lr = 0.9
    xs = GD(x_start,epochs,lr)#返回5次迭代后的列表
    print(xs)
    color = "r"#画红线
    plt.plot(xs,func(xs),c="b",label="lr={}".format(lr))#设置画板格式
    plt.scatter(xs,func(xs),c=color)#绘制散点图
    plt.legend()#用于设置图例
    plt.show()
if __name__ == "__main__":
    demo_GD()

