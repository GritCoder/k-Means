from numpy import *
def loadData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()#若split()中加分隔符'\t'，则转换浮点数失败,这一点需要注意一下
        dataMat.append([1.0,float(lineArr[0])])
        labelMat.append(float(lineArr[1]))
    return dataMat,labelMat
def gradAscent(dataMat, labelMat):
    dataMatrix = mat(dataMat)#转换成矩阵5X3
    labelMat = mat(labelMat).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001#初始步长
    maxCycles = 5#设置迭代次数
    theta = ones((n,1))#初始化系数矩阵
    for k in range(maxCycles):#开始迭代
        yPredict = dot(dataMatrix ,theta)#y的预测值
        error = labelMat - yPredict#计算预测值与真实值的误差
        theta = theta - alpha * dataMatrix.transpose() * error#迭代
    return theta#返回迭代后的系数矩阵
if __name__ == "__main__":
    dataMat,labelMat = loadData("gdTest.txt")
    #print(dataMat,labelMat)
    print(gradAscent(dataMat,labelMat))





#
# def Newton_3(c,t0):
#     t = t0
#     while abs(t ** 3 -c) > 1e-6:
#         t = t - (t ** 3 -c)/(3 * t ** 2)
#     return t
# print(Newton_3(2,1))
# def standRegress(xArr,yArr):#回归的过程中需要用到逆矩阵，该函数主要功能是判断矩阵是否可逆
#     xMat = mat(xArr); yMat = mat(yArr)#把二维列表转换成对应的矩阵
#     xTx = xMat.T*xMat
#     if float(linalg.det(xTx)) == 0.0:
#         print("this matrix is singular,can not inverse")
#         return
#     ws = xTx.I * (xMat.T * yMat)#根据平方误差最小化的公式推导求ws回归系数
#     #python3.6上面执行会报错，但是在2.7上面可以执行，这个bug有待解决
#     #ws = linalg.solve(xTx,xMat.T*yMat)
#     return ws

