import numpy as np
import time
'''
@function：创建一个实验样本
@#param: None
@return: 
    postingList [list] 词条切分后的文档集合
    classsVect [list] 一个类别标签的集合
'''
def loadDataSet():
    #实验文档样本
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #文档列表分类，1表示侮辱性，0表示非侮辱性
    return postingList, classVec

'''
@function：创建一个包含在所有文档中出现的不重复词的列表
@param: 
    dataSet 样本数据集
@return: 
    vocabSet [list] 不重复词条列表
'''
def createVocabList(dataSet):
    vocabSet = set([])      #创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #集合求并集
    return list(vocabSet)

'''
@function：词集模型，根据词汇表，将文档inputSet转化为向量
@param: 
    vocabList [list] 词汇表
    inputSet [list] 某个文档
@return: 
    returnVec [list] 对应文档向量
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #构建与词汇列表相同长度的列表
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    #如果词条在词汇表中存在，则相应位置标1
        else:print('the word: %s is not in my vocabulary' % word)
    return returnVec

'''
@function：朴素贝叶斯分类器训练函数
@param: 
    trainMatrix [list] 训练文档矩阵
    trainCtegory [list] 训练文档标签
@return: 
    p0Vect [list] 非侮辱性词条条件概率数组
    p1Vect [list] 侮辱性词条条件概率数组
    pAbusive [float] 文档属于侮辱类的概率
'''
def trainNB0(trainMatrix, trainCtegory):
    numTrainDoc = len(trainMatrix)  #训练样本个数
    pAbusive = sum(trainCtegory)/float(numTrainDoc) # 侮辱性文档的概率
    numWords = len(trainMatrix[0])
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords) #创建0数组,初始化概率
    p0Denum = 0.0; p1Denum = 0.0    #初始化分母
    for i in range(numTrainDoc):
        if trainCtegory[i] == 1: #侮辱性词条条件概率数组
            p1Num += trainMatrix[i]
            p1Denum += sum(trainMatrix[i])
        else:                   #非侮辱性词条条件概率数组
            p0Num += trainMatrix[i]
            p0Denum += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denum
    p0Vect = p0Num/p0Denum
    return p0Vect, p1Vect, pAbusive

'''
@function：朴素贝叶斯分类器训练函数(修改版)
@param: 
    trainMatrix [list] 训练文档矩阵
    trainCtegory [list] 训练文档标签
@return: 
    p0Vect [list] 非侮辱性词条条件概率数组
    p1Vect [list] 侮辱性词条条件概率数组
    pAbusive [float] 文档属于侮辱类的概率
'''
def trainNB(trainMatrix, trainCtegory):
    numTrainDoc = len(trainMatrix)  # 训练样本个数
    pAbusive = sum(trainCtegory) / float(numTrainDoc)  # 侮辱性文档的概率
    numWords = len(trainMatrix[0])
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)  # 创建单位数组,初始化概率
    p0Denum = 2.0;
    p1Denum = 2.0  # 初始化分母
    for i in range(numTrainDoc):
        if trainCtegory[i] == 1: #侮辱性词条条件概率数组
            p1Num += trainMatrix[i]
            p1Denum += sum(trainMatrix[i])
        else:                   #非侮辱性词条条件概率数组
            p0Num += trainMatrix[i]
            p0Denum += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denum)
    p0Vect = np.log(p0Num / p0Denum)
    return p0Vect, p1Vect, pAbusive

'''
@function：朴素贝叶斯的分类函数
@param: 
    vec2Classify [list] 要分类的向量
    p0Vect [list] 非侮辱性的条件概率数组
    p1Vect [list] 侮辱性的条件概率数组 
    pClass1 [list] 某一分类的概率（先验概率）
@return: 分类结果
'''
def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify*p1Vect) + np.log(pClass1)     #取对数
    p0 = sum(vec2Classify*p0Vect) + np.log(1- pClass1)  #取对数
    if p1 > p0:
        return 1
    else:
        return 0

'''
@function：算法测试
@param: None
@return: 测试结果
'''
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p1V, p0V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p1V, p0V, pAb))

'''
@function：词袋模型，根据词汇表，将文档inputSet转化为向量
@param: 
    vocabList [list] 词汇表
    inputSet [list] 某个文档
@return: 
    returnVec [list] 对应文档向量
'''
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1   #如果有单词出现，改为相应值增加1.
    return  returnVec

'''
@function：切分分本为字符串列表
@param: 
    bigString [string] 长本文
@return: 
    [list] 切分后的文本
'''
def textParse(bigString):
    import re
    listOfTokens = re.split('\\W*', bigString) #按非数字字母切换
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #词条长度>2，且用小写表示

'''
@function：文件解析和完整的垃圾邮件测试函数
@param: 
    bigString [string] 长本文
@return: 测试样本的错误率
'''
def spamTest():
    docList = []; classList = []; fullList = []
    for i in range(1,26):   #spam和ham样本分别为25
        wordList = textParse(open('./email/spam/%d.txt' % i).read())   #解析垃圾邮件
        docList.append(wordList)
        fullList.extend(wordList)
        classList.append(1)
        wordList = textParse(open('./email/ham/%d.txt' % i).read())  # 解析非邮件
        docList.append(wordList)
        fullList.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)    #创建词汇列表
    trainingSet = list(range(50)); testSet = [] #创建训练数据和测试数据,测试数据为10个
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet))) #随机取索引
        testSet.append(trainingSet[randIndex])  #测试集添加索引
        del(trainingSet[randIndex]) #删除训练集中的相应值
    trainMat = []; trainClasses = [] #创建训练矩阵和标签
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))   #将邮件词条转化为词条向量
        trainClasses.append(classList[docIndex])  #添加邮件类型输出标签
    p0V,p1V,pSam = trainNB(np.array(trainMat), np.array(trainClasses)) #朴素贝叶斯训练函数
    errorCount = 0
    for docIndex in testSet:    #对测试集进行测试
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSam) != classList[docIndex]: #判断朴素贝叶斯分类的正确性
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))

'''
@function：计算词频出现最高的前30个词
@param: 
    vocabList [list] 词汇表
    fullText [list] 统计的文本
@return: 
    sortedFreq [list] 词频最高的前30个词
'''
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}   #定义字典
    for token in vocabList:
        freqDict[token] = fullText.count(token) #字典存放个词条出现的个数
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) #对字典按值进行从大到小排序
    return sortedFreq[:30] #返回词频最高的前30个单词

'''
@function：RSS分类器测试函数
@param: 
    feed1 RSS源
    feed0 RSS源
@return: 
    vocabList [list] 词频最高的前30个词
    p0V [list]  出现0的条件概率数组 
    p1V [list]  出现1的条件概率数组
'''
def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries'])) #取词源中最小的
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) #构建词汇表
    top30Words = calcMostFreq(vocabList, fullText) #词频最高的前30个词
    for pairW in top30Words:
        if pairW in vocabList: vocabList.remove(pairW) #移除词频最高的前30个词
    trainingSet = list(range(2*minLen)); testSet=[] #构建训练集和测试集
    for i in range(20): #20个作为测试集
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = [] #训练矩阵和标签
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))   #将词源转换为词向量
        trainClasses.append(classList[docIndex])    #词向量对应的标签
    p0V,p1V,pSpam = trainNB(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet: #测试集错误率计算
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

'''
@function：计算词频出现最高的前30个词
@param: 
    ny 纽约RSS源
    sf 三藩市RSS源
@return: 按从高到低进行排序
'''
def getToWords(ny, sf):
    import operator
    vocabList,p0V,p1V = localWords(ny, sf)
    topNY = []; topSF = [] #构建列表
    for i in range(len(p0V)):
        if p0V[i] > -5.0: topSF.append((vocabList[i], p0V[i])) #对数概率>-6.0添加至列表
        if p1V[i] > -5.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)    #对列表元组概率从大到小排序
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF')
    for item in sortedSF: print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY')
    for item in sortedNY: print(item[0])

if __name__ == '__main__':
    start = time.clock()
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # returnVec = setOfWords2Vec(myVocabList, listOPosts[0])
    # print(returnVec)
    # returnVec = setOfWords2Vec(myVocabList, listOPosts[1])
    # print(returnVec)

    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print('trainMat:\n',trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # print('pAb:', pAb)

    # p0V, p1V, pAb = trainNB(trainMat, listClasses)
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # print('pAb:', pAb)
    # testingNB()

    # import re
    # mySent = 'This book is the best book on Python or M.L. I habe ever laid eyes upon.'
    # regEx = re.compile('\\W*') #非字母数字
    # listOfTockens = regEx.split(mySent) #切分得到词条列表，包括空格词条
    # print(listOfTockens)
    # listOfTockens = [tok.lower() for tok in listOfTockens if len(tok) > 0]  #去掉空字符串，并小写表示
    # print(listOfTockens)

    # spamTest()
    import feedparser
    ny = feedparser.parse('https://newyork.craigslist.org/search/pol?format=rss')
    sf = feedparser.parse('https://sfbay.craigslist.org/search/pol?format=rss')
    # vocabList, pSF, pNY = localWords(ny, sf)
    # vocabList, pSF, pNY = localWords(ny, sf)

    getToWords(ny,sf)
    end = time.clock()
    print('Finish in', end - start)

'''输出结果
the error rate is  0.45
SF**SF**SF**SF**SF**SF**SF**SF**SF**SF
liberal
years
not
just
...
app
help
with
law
want
since
america
NY**NY**NY**NY**NY**NY**NY**NY**NY**NY
paid
time
wmq
participants
...
study
qanon
631s
corrupt
00pm
midterms
political
Finish in 5.349823010491469
'''