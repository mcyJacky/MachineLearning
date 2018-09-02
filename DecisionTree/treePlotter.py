import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

decisionNode = dict(boxstyle='sawtooth', fc='0.8') #pad=0.3,tooth_size=None
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-') #head_length=0.4,head_width=0.2

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # xy 标注点
    # xytext 对标注点进行注释的点
    # axes fraction:  fraction of axes from lower left
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

# '''
# @function：绘制结点标注
# @#param:
#     nodeTxt [str] 标注文字
#     centerPt [xy] 注释文字位置的坐标
#     parentPt [xy] 标注对象点坐标
#     nodeType [dic] 标注字体框的格式
# @return: 标注后的图
# '''
# def createPlot():
#     fig = plt.figure(1, facecolor='white') #背景色
#     fig.clf()       #Clear the current figure
#     createPlot.ax1 = plt.subplot(111, frameon=False) ##用函数属性createPlot.ax1定义全局变量
#     plotNode('决策结点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('叶结点', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

'''
@function：获得叶子结点个数
@#param: 
    myTree [dict] 决策树
@return: 
    numLeafs [nums] 叶子结点个数
'''
def getNumLeafs(myTree):
    numLeafs = 0    #初始化叶子节点树
    firstStr = next(iter(myTree))   #当前树的第一个key
    secondDict = myTree[firstStr]   #当前树的第一个key值
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':    #如果该节点是判断节点，使用递归
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1   #如果是叶子节点，则叶子结点数+1
    return numLeafs

'''
@function：获得决策树层数
@#param: 
    myTree [dict] 决策树
@return: 
    maxDepth [nums] 树的层数
'''
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))   #当前树的第一个key
    secondDict = myTree[firstStr]   ##当前树的第一个key值
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':    #如果该节点是判断节点，使用递归
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

'''
@function：返回预定义的树结构(用于测试)
@#param: 
    i [num] 列表的索引
@return: 
    listOfTrees[i] [dict] 某决策树
'''
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

'''
@function：标注结点之间的判断文字
@#param: 
    cntrPt [xy] 标注文字的位置
    parentPt [xy] 标注对象的位置
    txtString [str] 标注文字
@return: 结点之间的判断文字
'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

'''
@function：绘制决策树方法
@#param: 
    myTree [dict] 待绘制的决策树对象
    parentPt [xy] 标注文字的位置
    nodeText [str] 标注文字
@return: 决策树图
'''
def plotTree(myTree, parentPt, nodeText):
    numLeafs = getNumLeafs(myTree)
    # depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff) #标注点的中心位置
    plotMidText(cntrPt, parentPt, nodeText)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #无坐标轴的图
    plotTree.totalW = float(getNumLeafs(inTree))    #全局变量定义树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))   #全局变量定义树的高度
    plotTree.xOff = -0.5/plotTree.totalW;plotTree.yOff = 1.0; #x,y偏移量
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == '__main__':
    # createPlot()
    myTree = retrieveTree(0)
    print(myTree)
    # numLeafs = getNumLeafs(myTree)
    # depth = getTreeDepth(myTree)
    # print('numLeafs: ', numLeafs)
    # print('depth: ', depth)

    createPlot(myTree)

'''输出结果
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
numLeafs:  3
depth:  2
'''