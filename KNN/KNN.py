import os
import numpy as np
import operator
import time
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

'''
#创建数据集和标签
#param: none
#return:
    array 数据集
    list  标签
'''
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
#功能：分类器
#param: 
    inX 输入向量
    dataSet 训练样本集
    labels 标签向量
    k 最近邻数目
#return: 分类标签
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]      # 训练样本矩阵行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 将输入向量inX进行维度扩充（使维度与dataSet相同），并作矩阵之差
    sqDiffMat = diffMat**2              # 将矩阵平方
    sqDistances = sqDiffMat.sum(axis=1) # 将矩阵元素按行相加: sum(axis=1)按行相加，sum(axis=0)按列相加
    distances = sqDistances**0.5        # 开根号，计算欧式距离
    sortedDistIndices = distances.argsort()     # 矩阵中的元素按从小到大进行排序后的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]   #距离最小前k个距离的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #从字典中统计标签的个数
    # 将字典分解为元组列表[('A', 2), ('B', 1)]
    # 将元组列表标签个数按值value(用key=operator.itemgetter(1))进行从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
#功能：解析文本文件
#param: 
    fileName 文件名称txt
#return: 
    returnMat [matrix] 训练样本矩阵
    classLabelVector [list] 类标签向量
'''
def file2Mmatrix(filename):
    dict = {'didntLike':1, 'smallDoses':2, 'largeDoses':3}
    fr = open(filename)             #打开txt文本文件
    arrayOLines = fr.readlines()    #读取所有行，返回行列表
    returnMat = np.zeros((len(arrayOLines), 3)) #准备返回样本矩阵
    classLabelVector = []           #准备返回的类标签向量
    index = 0
    for line in arrayOLines:
        line = line.strip()         #截取头尾的所有回车字符
        listFromLine = line.split('\t')     #使用tab字符\t将行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1].isdigit():
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(dict.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

'''
#功能：可视化训练样本
#param: 
    datingDataMat 训练样本矩阵
    datingLabels 类标签向量 
#return: plot图
'''
def viewDatas(datingDataMat, datingLabels):
    LabelColors = []
    for i in datingLabels:
        if i == 1:
            LabelColors.append('red')
        elif i == 2:
            LabelColors.append('green')
        elif i == 3:
            LabelColors.append('blue')
    fig = plt.figure(figsize=(10, 8))
    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(212)
    #散点图1：以矩阵第2列，第3列绘制图，散点大小为15，透明度为0.5
    ax0.scatter(datingDataMat[:,1], datingDataMat[:,2], s=15, color=LabelColors, alpha=0.5)
    # 设置标题title, 标签label
    axs0_title = ax0.set_title(u"玩视频游戏所消耗时间占比与每周消费的冰激淋公升数")
    axs0_xlabel = ax0.set_xlabel(u"玩视频游戏所消耗时间占比")
    axs0_ylabel = ax0.set_ylabel(u"每周消费的冰激淋公升数")
    # 设置相应属性
    plt.setp(axs0_title, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel, size=8, weight='bold', color='black')
    plt.setp(axs0_ylabel, size=8, weight='bold', color='black')

    # 散点图2：以矩阵第1列，第2列绘制图，散点大小为15，透明度为0.5
    ax1.scatter(datingDataMat[:, 0], datingDataMat[:, 1], s=15, c=LabelColors, alpha=0.5)
    # 设置标题title, 标签label
    axs1_title = ax1.set_title(u"每年获得的飞行常客里程数与玩视频游戏所消耗时间占比")
    axs1_xlabel = ax1.set_xlabel(u"每年获得的飞行常客里程数")
    axs1_ylabel = ax1.set_ylabel(u"玩视频游戏所消耗时间占比")
    # 设置相应属性
    plt.setp(axs1_title, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel, size=8, weight='bold', color='black')
    plt.setp(axs1_ylabel, size=8, weight='bold', color='black')

    # 散点图3：以矩阵第1列，第3列绘制图，散点大小为15，透明度为0.5
    ax2.scatter(datingDataMat[:, 0], datingDataMat[:, 2], s=15, c=LabelColors, alpha=0.5)
    # 设置标题title, 标签label
    axs2_title = ax2.set_title(u"每年获得的飞行常客里程数与每周消费的冰激淋公升数")
    axs2_xlabel = ax2.set_xlabel(u"每年获得的飞行常客里程数")
    axs2_ylabel = ax2.set_ylabel(u"每周消费的冰激淋公升数")
    # 设置相应属性
    plt.setp(axs2_title, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel, size=8, weight='bold', color='black')
    plt.setp(axs2_ylabel, size=8, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='red', marker='.',
                              markersize=6, label=u'不喜欢')
    smallDoses = mlines.Line2D([], [], color='green', marker='.',
                               markersize=6, label=u'魅力一般')
    largeDoses = mlines.Line2D([], [], color='blue', marker='.',
                               markersize=6, label=u'极具魅力')
    # 添加图例
    ax0.legend(handles=[didntLike, smallDoses, largeDoses])
    ax1.legend(handles=[didntLike, smallDoses, largeDoses])
    ax2.legend(handles=[didntLike, smallDoses, largeDoses])
    plt.show()

'''
#功能：归一化数值
#param: 
    dataSet 训练样本矩阵
#return: 
    normDataSet 归一化矩阵
    ranges 训练样本极值之差矩阵
    minVals 训练样本组成最小值矩阵
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)    #选取列最小值
    maxVals = dataSet.max(0)    #选取列最大值
    ranges = maxVals - minVals  #最大值与最小值之差
    normDataSet = np.zeros(np.shape(dataSet))   #定义归一化零矩阵
    m = dataSet.shape[0]    #行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))    #原样本矩阵减去矩阵列中的最小值
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   #矩阵归一化
    return  normDataSet, ranges, minVals

'''
#功能：分类器算法测试
#param: None
#return: 测试结果
'''
def datingClassTest():
    hoRatio = 0.10  #测试数据为样本的10%
    fileName = 'datingTestSet.txt'
    datingDataMat, datingLabels = file2Mmatrix(fileName)    #样本格式转换
    normMat, ranges, minVals = autoNorm(datingDataMat)      #样本数值归一化
    m = normMat.shape[0]            #行数
    numTestVecs = int(m*hoRatio)    #测试样本的个数
    errorCount = 0.0                #统计错误结果个数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

'''
#功能：约会网站预测函数
#param: None
#return: 预测结果
'''
def classifyPerson():
    resultList = ['讨厌', '一般喜欢', '非常喜欢']
    precentTats = float(input("玩视频游戏消耗时间百分比:"))
    ffMiles = float(input("每年获得飞行的常客里程数:"))
    iceCream = float(input("每周消耗的冰淇淋公升数:"))
    datingDataMat, datingLabels = file2Mmatrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, precentTats, iceCream])  #输入向量
    normInArr = (inArr - minVals)/ranges    #输入向量归一化数值
    classifierResult = classify0(normInArr, normMat, datingLabels, 3)
    print('u probaly like this person: '  + resultList[classifierResult - 1])

'''
#功能：将图片矩阵格式(32,32)转化为向量格式(1,1024)
#param: 
    filename 文件名
#return: 
    returnVect 图片格式数组
'''
def img2Vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        listStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(listStr[j])
    return  returnVect

''' 
#功能：手写数字识别系统的测试代码
#param: Nonee
#return: 测试结果
'''
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('./trainingDigits')   #获取文件夹下的文件
    m = len(trainingFileList)   #文件的个数
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]       #第i个文件名
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])#将文件按'_'分割，获取矩阵实际标签
        hwLabels.append(classNumStr)            #添加标签
        trainingMat[i,:] = img2Vector('./trainingDigits/' + fileNameStr)    #格式转换
    testFileList = os.listdir('./testDigits')   #测试文件
    errorCount = 0.0            #预测错误个数
    mTest = len(testFileList)   #测试数据数量
    for i in range(mTest):
        fileNameStr = testFileList[i]  # 第i个文件名
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])  # 将文件按'_'分割，获取矩阵实际标签
        vectorUnderTest= img2Vector('./testDigits/' + fileNameStr)  # 格式转换
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe rotal error rate is: %f' % (errorCount/float(mTest)))

if __name__ == '__main__':
    start = time.clock()
    # test = [1, 0.8]
    # group, labels = createDataSet()
    # test_class = classify0(test, group, labels, 3)
    # print(test_class)

    # fileName = 'datingTestSet.txt'
    # datingDataMat, datingLabels = file2Mmatrix(fileName)
    # print(datingDataMat)
    # print(datingLabels[0:20])

    # viewDatas(datingDataMat, datingLabels)

    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # print(normMat)
    # print('--------')
    # print(ranges)
    # print('--------')
    # print(minVals)

    # datingClassTest()

    # classifyPerson()

    # testVect = img2Vector('0_0.txt')
    # print(testVect[0, 0:20])

    handwritingClassTest()
    end = time.clock()
    print('Finished in', end - start)

'''输出结果：
...
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9

the total number of errors is: 10

the rotal error rate is: 0.010571
Finished in 34.70805667447233
'''