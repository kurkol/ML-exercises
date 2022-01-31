import matplotlib.pyplot as plt
import numpy as np

"""
说明:利用sigmoid函数分类

"""
def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                        #创建分类列表
    fr = open('testSet.txt')                                            #打开文件
    for line in fr.readlines():                                  
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])        #添加数据，前面加个1是为了方便矩阵运算
        labelMat.append(int(lineArr[2]))                                #添加分类
    fr.close()                                                            
    return dataMat, labelMat
#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# 梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat，x0=1,x1=第一列数据，x2=第二列数据
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n,1))                                             #设置w^T的初始值为（1,1,1）
    
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    print(weights)
    return weights.getA()                                                #将矩阵转换为数组，返回权重数组

#画图
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x1 = np.arange(-3.0, 3.0, 0.001)                                     #生成步长为0.001的X值
    x2 = (-weights[0] - weights[1] * x1) / weights[2]                     #根据w0*x0+w1*x1+w2*x2=0获得x2的值
    ax.plot(x1, x2)                                                      #根据x1和x2画出分类直线
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)
