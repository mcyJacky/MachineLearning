import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

'''
#function: 载入训练数据集
#param: none
#return:
    dataMat [list] 特征向量
    labelMat [list] 特征标签
'''
def loadDataSet():
    dataMat = []; labelMat = [] #初始化特征向量和分类标签列表
    with open(r'.\testSet.txt', 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1, float(lineArr[0]), float(lineArr[1])])   #插入特征值，前面多加1
            labelMat.append(int(lineArr[-1]))   #插入分类标签
    fr.close()
    return dataMat, labelMat

'''
#function: sigmoid函数
#param: 
    inX [matrix/num] 函数参数
#return: 函数值
'''
def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))

'''
#function: 梯度上升法
#param: 
    dataMatIn [list] 特征向量
    classLabels [list] 特征标签
#return: 
    weights [matrix] 回归系数
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)              #列表转化为矩阵
    m, n = np.shape(dataMatrix)
    labelMat = np.mat(classLabels).transpose()  #矩阵的转置
    alpha = 0.001       #迭代步长
    maxCycles = 500     #迭代次数
    weights = np.ones((n,1))    #初始化回归系数
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   #预测值
        error = labelMat - h    #误差
        weights = weights + alpha * dataMatrix.transpose() * error  #梯度上升
    return weights

'''
#function: 绘制拟合线
#param: 
    weights [matrix] 回归系数
#return: 拟合曲线图
'''
def plotBestFit(weights):
    dataMat, lableMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] #行数
    xcord1 = []; ycord1 = [] #分类点1
    xcord2 = []; ycord2 = [] #分类点2
    for i in range(n):
        if int(lableMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append([dataArr[i, 2]])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append([dataArr[i, 2]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') #绘制散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x1 = np.arange(-3.0, 3.0, 0.1) #x轴的取值范围
    weights = np.array(weights).ravel() #将数组降维
    x2 = (-weights[0]- weights[1] * x1) / weights[2] #y = w0 + w1*x + w2 * x ==> 当y = 0时
    ax.plot(x1, x2)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.show()

'''
#function: 随机梯度上升
#param: 
    dataMatrix [numpy.ndarray] 特征向量
    classLabels [list] 特征标签
#return: 
    weights [matrix] 回归系数
'''
def stocGradAscent0(dataMatrix, classLabel):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n) #创建一维数组 n表示数组个数
    weights_changes = np.array([])
    for i in range(m):
        h = sigmoid(np.sum(dataMatrix[i] * weights)) #数值
        error = classLabel[i] - h   #误差值，非向量
        weights = weights + alpha * error * dataMatrix[i] #梯度上升
    return weights

'''
#function: 改进的随机梯度上升
#param: 
    dataMatrix [numpy.ndarray] 特征向量
    classLabels [list] 特征标签
    numIter [num] 迭代次数
#return: 
    weights [numpy.ndarray] 回归系数
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01  # 调整alpha步长
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # 选取随机量进行更新
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

'''
#function: 分类函数
#param: 
    inX [numpy.ndarray] 测试特征向量
    weights [numpy.ndarray] 回归系数
#return: 分类结果
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

'''
#function: 数据处理主函数
#param: None
#return: 
    errorRate [num] 测试数据的错误率
'''
def colicTest():
    frTrain = open(r'.\horseColicTraining.txt') #打开训练文本文件
    frTest = open(r'.\horseColicTest.txt')      #打开测试文本文件
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)                 #添加训练特征向量
        trainingLabels.append(float(currLine[-1]))  #添加训练特征标签
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500) #使用改进的随机梯度算法计算回归系数
    errorCount = 0; numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]): #分类判断
            errorCount += 1
    errorRate = float(errorCount) / numTestVect #错误率计算
    print(f"the error rate of this test is: {errorRate}")
    return errorRate

'''
#function: 多次计算求平均值
#param: None
#return: 平均错误率
'''
def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print(f'after {numTests} iterations the avg error rate is: {errorSum/float(numTests)}')

if __name__ == '__main__':
    start = time.clock()
    # dataArr, labelMat = loadDataSet()
    # print(dataArr)
    # print(labelMat)
    #
    # weights = gradAscent(dataArr, labelMat)
    # print(weights)
    # plotBestFit(weights)

    # weights = stocGradAscent0(np.array(dataArr), labelMat)
    # print(weights)
    # plotBestFit(weights)

    # weights = stocGradAscent1(np.array(dataArr), labelMat)
    # print(weights)
    # plotBestFit(weights)
    multiTest()
    end = time.clock()
    print(end - start)

'''输出结果：
the error rate of this test is: 0.34328358208955223
the error rate of this test is: 0.23880597014925373
the error rate of this test is: 0.34328358208955223
the error rate of this test is: 0.40298507462686567
the error rate of this test is: 0.31343283582089554
the error rate of this test is: 0.44776119402985076
the error rate of this test is: 0.3582089552238806
the error rate of this test is: 0.417910447761194
the error rate of this test is: 0.3582089552238806
the error rate of this test is: 0.3880597014925373
after 10 iterations the avg error rate is: 0.3611940298507463
29.20000236854104
'''