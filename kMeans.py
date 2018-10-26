from numpy import *
import warnings
warnings.filterwarnings("ignore")#忽略运行中的警告信息
def loadData(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #fltLine = map(float,curLine)#映射成浮点数,2.7中可以执行 python3.6中执行失败
        fltLine = [float(f) for f in curLine]#python3.6中转换方式，利用到了列表表达式
        dataMat.append(fltLine)
    dataMat = mat(dataMat)
    return dataMat
def distance(vecA,vecB):#注意输入的是向量，挺神奇，该函数也可以直接计算向量的欧氏距离
    return sqrt(sum(power(vecA - vecB,2)))#计算欧氏距离
def randCent(dataSet,k):
    n = shape(dataSet)[1]#获取列
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j] - minJ))
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)#通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
    return centroids
def kMeans(dataMat,k,dist=distance,createCent=randCent):
    m = shape(dataMat)[0]#获取行数
    cluster = mat(zeros((m,2)))#初始化簇分配结果矩阵 包括索引和误差(即该点到簇质心的距离平方值)
    centroids = createCent(dataMat,k)#随机选取簇矩阵
    clusterChanged = True#簇是否改变的标志
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#对于每一行数据
            minDist = inf ; minIndex = -1#初始化最小值和对应的索引
            for j in range(k):#对于每一行簇,计算到数据点的距离，注意，这里的每一个数据点是指的向量
                distJI = dist(centroids[j,:],dataMat[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if cluster[i,0] != minIndex:
                clusterChanged = True
            cluster[i,:] = minIndex,minDist ** 2#一开始有bug 是因为这句代码忘记加了，要仔细！！！
        #print(centroids)
        for cent in range(k):
            ptsInCluster = dataMat[nonzero(cluster[:,0].A == cent)[0]]#矩阵名.A表示将矩阵转化成列表类型，这里是二维的，0表示取对应的行数据
            #print(nonzero(cluster[:,0].A == cent)[0])#nonzero在这里使用是一个技巧 通过输出调试和相关博客理解了含义
            #print(ptsInCluster) #简言之就是返回值不为0元素的下标，将一个布尔类型的数组转换成整数数组(一维)或者元组(二维)
            centroids[cent,:] = mean(ptsInCluster,axis=0)#按列方向进行计算均值
        return centroids,cluster#最后返回质心与所有的类分配结果
def main():
    dataMat = loadData("testSet.txt")
    #print(dataMat)
    #print(min(dataMat[:,0]))
    #print(randCent(dataMat,4))
    #print(dataMat[0],dataMat[1])
    #print(distance(dataMat[0],dataMat[1]))
    myCentroids,cluster = kMeans(dataMat,4)
    print(myCentroids)
    print(cluster)
if __name__ == "__main__":
    main()