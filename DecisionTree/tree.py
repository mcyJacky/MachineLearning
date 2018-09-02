'''
Created on Aug 26, 2018
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author:mcyJacky
'''
import time
import math
import operator

'''
@function：创建一个训练数据集
@#param: None
@return: 
    dataSet [list] 训练数据集
    labels [list] 特征标签
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''
@function：给定数据集熵的计算
@#param: dataSet [list] 训练数据集列表
@return: 
    shannonEnt [num] 熵(香农熵)
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}    #定义类标签数量字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    for key in labelCounts: #进行熵计算
        prob = float(labelCounts[key])/numEntries
        shannoEnt -= prob * math.log(prob, 2)
    return shannoEnt

'''
@function：划分数据集
@#param: 
    dataSet [list] 待划分训练数据集
    axis 划分数据集的特征(索引)
    value 需要返回特征的值
@return: 
    retDataSet [list] 划分后的数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
@function：通过信息增益选择最优特征
@#param: 
    dataSet [list] 待划分训练数据集
@return: 
    bestFeature [num] 最优特征位于列表的索引
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #特征个数
    baseEntropy = calcShannonEnt(dataSet) #熵的结果
    bestInfoGain = 0.0; bestFeature = -1    #初始化信息增益和最优特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #某一特征所有取值列表
        uniqueValues =set(featList) #用集合去取列表中相同部分
        newEntropy = 0.0        #初始化特征熵
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)    #H划分特征数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i #最优特征索引
    return bestFeature

'''
@function：通过多数表决返回次数分类最多的名称
@#param: 
    classList [list] 类标签
@return: 
    sortedClassCount[0][0] 次数最多的类标签的名称 
'''
def majorityCnt(classList):
    classCount = {}     #定义初始化分类标签字典存储变量
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True) #从大到小排序
    return sortedClassCount[0][0]

'''
@function：构建决策树
@#param: 
    dataSet [list] 待划分训练数据集
    labels []   特征标签
@return: 
    myTree [dict] 决策树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]    #分类标签列表
    if classList.count(classList[0]) == len(classList): #类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:       #使用完所有特征，仍然不能将数据集划分成仅包含唯一类别的分组，返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #最优特征索引
    bestFeatLabel = labels[bestFeat] #最优特征的名称
    myTree = {bestFeatLabel:{}}     #初始化决策树
    del(labels[bestFeat] )
    featValues = [example[bestFeat] for example in dataSet] #最优特征中特征值列表
    uniqueVals = set(featValues) #特征值列表去重
    for value in uniqueVals:
        subLables = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables) #递归
    return myTree

'''
@function：决策树的分类函数
@#param: 
    inputTree [dict] 决策树
    featLabels [list] 特征标签
    testVec [list] 测试对象
@return: 
    classLabel 分类标签
'''
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

'''
@function：序列化对象存储决策树
@#param: 
    inputTree [dict] 决策树
    filename 存储后的文件
@return: None
'''
def storeTree(inputTree, filename):
    import pickle   #导入pickle模块
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)  #序列化
    fw.close()

'''
@function：将对象进行反序列化（读取决策树对象）
@#param: 
    filename 存储后的文件
@return: 决策树模型
'''
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)  #反序列化

if __name__ == '__main__':
    start = time.clock()
    myDat, labels = createDataSet()
    print(myDat)
    # print(labels)

    # entropy = calcShannonEnt(myDat)
    # print("entropy: ", entropy)

    # retData = splitDataSet(myDat, 0, 1)
    # print(retData)
    # bestFeat = chooseBestFeatureToSplit(myDat)
    # print('bestFeat', bestFeat)

    # classList = ['yes', 'yes', 'yse', 'no']
    # maxLabel = majorityCnt(classList)
    # print(maxLabel)

    # myTree = createTree(myDat, labels)
    # print(myTree)

    import treePlotter as tp
    myTree = tp.retrieveTree(0)
    # result = classify(myTree, labels, [1,0])
    # print('result: ', result)

    # filename = './storageTree.txt' #当前目录下的文件
    # storeTree(myTree, filename)
    # rTree = grabTree(filename)
    # print('rTree', rTree)

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 年龄， 散光，
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    tp.createPlot(lensesTree)

    end = time.clock()
    print(end - start)

'''输出结果
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
{'tearRate': {'reduced': 'no lenses', 'normal': {'astigmatic': {'yes': {'prescript': {'hyper': {'age': {'young': 'hard', 'pre': 'no lenses', 'presbyopic': 'no lenses'}}, 'myope': 'hard'}}, 'no': {'age': {'young': 'soft', 'pre': 'soft', 'presbyopic': {'prescript': {'hyper': 'soft', 'myope': 'no lenses'}}}}}}}}
'''