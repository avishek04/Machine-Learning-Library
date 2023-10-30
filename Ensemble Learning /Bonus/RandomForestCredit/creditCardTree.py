import math
import random
import matplotlib.pyplot as plt
import numpy as np

#Common Code --------------------------------------------- Start

featAttrDict = {
    "balance": {
        "low": 0,
        "med": 0,
        "high": 0,
    },
    "sex": {
        "1": 0,
        "2": 0
    },
    "education": {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0
    },
    "marriage": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0
    },
    "age": {
        "low": 0,
        "med": 0,
        "high": 0
    },
    "pay1": {
        "-2": 0,
        "-1": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    },
    "pay2": {
        "-2": 0,
        "-1": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    },
    "pay3": {
        "-2": 0,
        "-1": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    },
    "pay4": {
        "-2": 0,
        "-1": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    },
    "pay5": {
        "-2": 0,
        "-1": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    },
    "pay6": {
        "-2": 0,
        "-1": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    },
    "billAmt1": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "billAmt2": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
   "billAmt3": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "billAmt4": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "billAmt5": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "billAmt6": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "payAmt1": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "payAmt2": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
   "payAmt3": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "payAmt4": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "payAmt5": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "payAmt6": {
        "low": 0,
        "med" : 0,
        "high": 0
    },
    "y": {
        "0": 0,
        "1": 0
    }
}

dataList = []
with open('creditCardTrain.csv', 'r') as f:
    next(f)
    for line in f:
        terms = line.strip().split(',')
        dataList.append(terms)

random.shuffle(dataList)
arr = np.array(dataList[:24000])
testArr = np.array(dataList[24000:])

class Node:
   def __init__(self, name, depth, leaf, nodes):
      self.name = name
      self.depth = depth
      self.isLeaf = leaf
      self.children = nodes
      

def resetFeatureAttributeDictionary():
    for feature in featAttrDict:
        for attribute in featAttrDict[feature]:
            featAttrDict[feature][attribute] = 0


def setFeatureAttrCount(dataset, col):
    resetFeatureAttributeDictionary()
    if len(col) > 1:
        for line in dataset:
            i = 0
            for colName in col:
                featAttrDict[colName][line[i]] += 1
                i += 1


def setContiniousValue():
    for line in arr:
        if (int(line[0]) < 100000):
            line[0] = "low"
        elif (int(line[0]) < 300000):
            line[0] = "med" 
        else:
            line[0] = "high"

        if (int(line[2]) > 3 or int(line[2]) < 1):
            line[2] = "4"

        if (int(line[3]) > 2):
            line[3] = "3"

        if (int(line[4]) < 30):
            line[4] = "low"
        elif (int(line[11]) < 50):
            line[4] = "med" 
        else:
            line[4] = "high"

        if (int(line[11]) < 10000):
            line[11] = "low"
        elif (int(line[11]) < 100000):
            line[11] = "med" 
        else:
            line[11] = "high"
        
        if (int(line[12]) < 10000):
            line[12] = "low"
        elif (int(line[12]) < 100000):
            line[12] = "med" 
        else:
            line[12] = "high"

        if (int(line[13]) < 10000):
            line[13] = "low"
        elif (int(line[13]) < 100000):
            line[13] = "med" 
        else:
            line[13] = "high"

        if (int(line[14]) < 10000):
            line[14] = "low"
        elif (int(line[14]) < 100000):
            line[14] = "med" 
        else:
            line[14] = "high"

        if (int(line[15]) < 10000):
            line[15] = "low"
        elif (int(line[15]) < 100000):
            line[15] = "med" 
        else:
            line[15] = "high"

        if (int(line[16]) < 10000):
            line[16] = "low"
        elif (int(line[16]) < 100000):
            line[16] = "med" 
        else:
            line[16] = "high"

        if (int(line[17]) < 3000):
            line[17] = "low"
        elif (int(line[17]) < 10000):
            line[17] = "med"
        else:
            line[17] = "high"

        if (int(line[18]) < 3000):
            line[18] = "low"
        elif (int(line[18]) < 10000):
            line[18] = "med"
        else:
            line[18] = "high"

        if (int(line[19]) < 3000):
            line[19] = "low"
        elif (int(line[19]) < 10000):
            line[19] = "med"
        else:
            line[19] = "high"

        if (int(line[20]) < 3000):
            line[20] = "low"
        elif (int(line[20]) < 10000):
            line[20] = "med"
        else:
            line[20] = "high"

        if (int(line[21]) < 3000):
            line[21] = "low"
        elif (int(line[21]) < 10000):
            line[21] = "med"
        else:
            line[21] = "high"

        if (int(line[22]) < 3000):
            line[22] = "low"
        elif (int(line[22]) < 10000):
            line[22] = "med"
        else:
            line[22] = "high"


def setContiniousValueTest():
    for line in testArr:
        if (int(line[0]) < 100000):
            line[0] = "low"
        elif (int(line[11]) < 300000):
            line[0] = "med" 
        else:
            line[0] = "high"

        if (int(line[2]) > 3 or int(line[2]) < 1):
            line[2] = "4"

        if (int(line[3]) > 2):
            line[3] = "3"

        if (int(line[4]) < 30):
            line[4] = "low"
        elif (int(line[11]) < 50):
            line[4] = "med" 
        else:
            line[4] = "high"

        if (int(line[11]) < 10000):
            line[11] = "low"
        elif (int(line[11]) < 100000):
            line[11] = "med" 
        else:
            line[11] = "high"
        
        if (int(line[12]) < 10000):
            line[12] = "low"
        elif (int(line[12]) < 100000):
            line[12] = "med" 
        else:
            line[12] = "high"

        if (int(line[13]) < 10000):
            line[13] = "low"
        elif (int(line[13]) < 100000):
            line[13] = "med" 
        else:
            line[13] = "high"

        if (int(line[14]) < 10000):
            line[14] = "low"
        elif (int(line[14]) < 100000):
            line[14] = "med" 
        else:
            line[14] = "high"

        if (int(line[15]) < 10000):
            line[15] = "low"
        elif (int(line[15]) < 100000):
            line[15] = "med" 
        else:
            line[15] = "high"

        if (int(line[16]) < 10000):
            line[16] = "low"
        elif (int(line[16]) < 100000):
            line[16] = "med" 
        else:
            line[16] = "high"

        if (int(line[17]) < 3000):
            line[17] = "low"
        elif (int(line[17]) < 10000):
            line[17] = "med"
        else:
            line[17] = "high"

        if (int(line[18]) < 3000):
            line[18] = "low"
        elif (int(line[18]) < 10000):
            line[18] = "med"
        else:
            line[18] = "high"

        if (int(line[19]) < 3000):
            line[19] = "low"
        elif (int(line[19]) < 10000):
            line[19] = "med"
        else:
            line[19] = "high"

        if (int(line[20]) < 3000):
            line[20] = "low"
        elif (int(line[20]) < 10000):
            line[20] = "med"
        else:
            line[20] = "high"

        if (int(line[21]) < 3000):
            line[21] = "low"
        elif (int(line[21]) < 10000):
            line[21] = "med"
        else:
            line[21] = "high"

        if (int(line[22]) < 3000):
            line[22] = "low"
        elif (int(line[22]) < 10000):
            line[22] = "med"
        else:
            line[22] = "high"

#Common Code --------------------------------------------- End


#Entropy Calculation -------------------------------------- Start

def overallEntropy(dataset):
    totalCount = len(dataset)
    labelCount = 0
    entropy = 0

    for label in featAttrDict["y"]:
        if featAttrDict["y"][label] > 0:
            labelCount += 1

    if labelCount > 1:
        for label in featAttrDict["y"]:
            calc = featAttrDict["y"][label]/(totalCount * 1.0)
            prob = calc if calc > 0.0 else 1
            entropy += (prob) * math.log(prob, 2)
    return -entropy   


def attributeEntropy(dataset, attribute):
    entropy = 0
    attrLabelDict = {
        "0": 0,
        "1": 0,
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    if count > 0:
        prob1 = (attrLabelDict["0"]/count) if (attrLabelDict["0"]/count) > 0.0 else 1
        prob2 = (attrLabelDict["1"]/count) if (attrLabelDict["1"]/count) > 0.0 else 1
        entropy = - prob1 * math.log(prob1, 2) - prob2 * math.log(prob2, 2)

    return entropy


def featureEntropy(data, featureName):
    entropy = 0
    totalSize = len(data)*1.0
    for attribute in featAttrDict[featureName]:
        if featAttrDict[featureName][attribute] > 0:
            entropy += (featAttrDict[featureName][attribute]/totalSize) * attributeEntropy(data, attribute)
    return entropy


def findLowestEntropyFeature(dataset, colNameArr):
    minEntropy = 2
    colLen = len(colNameArr) - 1
    pickCount = 6 if 6 < colLen else colLen
    newNameArr = colNameArr.copy().tolist()
    newNameArr.pop()
    subColArr = random.sample(newNameArr, pickCount)
    minEntropyCol = subColArr[0]
    
    totalCol = len(dataset[0])
    for feature in subColArr:
        colCount = newNameArr.index(feature)
        featureLabelCol = dataset[:, [colCount, totalCol - 1]]
        entropy = featureEntropy(featureLabelCol, feature)
        if entropy < minEntropy:
            minEntropy = entropy
            minEntropyCol = feature
    return minEntropyCol

#Entropy Calculation -------------------------------------- End

#ID3 Recursion ------------------------------------------- Start

def ID3RandTreeLearn(dataset, colNames, depth, maxDepth):
    setFeatureAttrCount(dataset, colNames)
    H = overallEntropy(dataset)
    if H == 0:
        return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
    
    colWithLowestEntropy = findLowestEntropyFeature(dataset, colNames)
    colPos = np.where(colNames == colWithLowestEntropy)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestEntropy]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            children[branch] = Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
        else:
            trimDataset = np.delete(trimDataset, colPos, 1)
            children[branch] = ID3RandTreeLearn(trimDataset, colNames, depth+1, maxDepth)

    return Node(colWithLowestEntropy, depth, False, children)

#ID3 Recursion ------------------------------------------- End


#Testing - Traversing the tree --------------------------- Start

def TestPredict(testVector, node, colArr):
    if node.isLeaf:
        return node.children
    
    colIndex = 0
    for feature in colArr:
        if feature == node.name:
            nextNode = node.children[testVector[colIndex]]
            newTestVector = np.delete(testVector, colIndex)
            newColArr = np.delete(colArr, colIndex)
            return TestPredict(newTestVector, nextNode, newColArr)
        colIndex += 1
     

def GetPrediction(rootNode, line):
    return TestPredict(line, rootNode, np.array(["balance", "sex", "education", "marriage", "age", "pay1", "pay2", "pay3", "pay4", "pay5", "pay6", "billAmt1", "billAmt2", "billAmt3", "billAmt4", "billAmt5", "billAmt6", "payAmt1", "payAmt2", "payAmt3", "payAmt4", "payAmt5", "payAmt6"]))

treeList = []
#Driver Function -------------------------- Start

def DecisionDriver():
    #Change the max depth value here to change the depth of the decision tree.
    maxDepth = 20
    T = 500
    samplePick = 1000
    # train_error_List = []
    # test_error_List = []
    # dataLen = 24000
    # testDataLen = len(testArr)

    for i in range(T):
        newArr = np.array(random.choices(arr, k = samplePick))
        node = ID3RandTreeLearn(newArr, np.array(["balance", "sex", "education", "marriage", "age", "pay1", "pay2", "pay3", "pay4", "pay5", "pay6", "billAmt1", "billAmt2", "billAmt3", "billAmt4", "billAmt5", "billAmt6", "payAmt1", "payAmt2", "payAmt3", "payAmt4", "payAmt5", "payAmt6", "y"]), 0, maxDepth)
        treeList.append(node)

    #     last = len(arr[0]) - 1
    #     predictResults = []
    #     testPredictResults = []

    #     for line in arr:
    #         sum = 0
    #         for node in treeList:
    #             sum += 1 if GetPrediction(node, line) == "1" else -1
    #         res = "1" if sum >= 0 else "0"
    #         predictResults.append(res)

    #     for line in testArr:
    #         sum = 0
    #         for node in treeList:
    #             sum += 1 if GetPrediction(node, line) == "1" else -1
    #         res = "1" if sum >= 0 else "0"
    #         testPredictResults.append(res)

    #     count = 0
    #     errorCount = 0
    #     for line in arr:
    #         if predictResults[count] != line[last]:
    #             errorCount += 1
    #         count += 1

    #     testCount = 0
    #     testErrorCount = 0
    #     for line in testArr:
    #         if testPredictResults[testCount] != line[last]:
    #             testErrorCount += 1
    #         testCount += 1

    #     train_error_List.append(errorCount/dataLen)
    #     test_error_List.append(testErrorCount/testDataLen)

    # return train_error_List, test_error_List
    

setContiniousValue()
setContiniousValueTest()
DecisionDriver()

 #rows = len(arr) #col = 500
#training error report
arrLen = len(arr)
trainPredictionList = np.zeros(shape = (arrLen, 500))
trainPrediction = [['' for x in range(500)] for y in range(arrLen)]
train_errors = []

count = 0
firstNode = treeList[0]
for line in arr:
    trainPredictionList[count][0] = 1 if GetPrediction(firstNode, line) == "1" else -1
    count += 1

treeCount = 1
for node in treeList[1:]:
    rowCount = 0
    for line in arr:
        predictionVal = 1 if GetPrediction(node, line) == "1" else -1
        trainPredictionList[rowCount][treeCount] = trainPredictionList[rowCount][treeCount - 1] + predictionVal
        rowCount += 1
    treeCount += 1

for i in range(arrLen):
    for j in range(500):
        trainPrediction[i][j] = "1" if trainPredictionList[i][j] >= 0 else "0"

lastIndx = len(arr[0]) - 1
for i in range(500):
    errorCount = 0
    for j in range(arrLen):
        if arr[j][lastIndx] != trainPrediction[j][i]:
            errorCount += 1
    train_errors.append(errorCount/arrLen)


#test error report
testArrLen = len(testArr)
testPredictionList = np.zeros(shape = (testArrLen, 500))
testPrediction = [['' for x in range(500)] for y in range(testArrLen)]
test_errors = []

testCount = 0
firstTestNode = treeList[0]
for line in testArr:
    testPredictionList[testCount][0] = 1 if GetPrediction(firstTestNode, line) == "1" else -1
    testCount += 1

treeTestCount = 1
for node in treeList[1:]:
    rowCount = 0
    for line in testArr:
        predictionVal = 1 if GetPrediction(node, line) == "1" else -1
        testPredictionList[rowCount][treeTestCount] = testPredictionList[rowCount][treeTestCount - 1] + predictionVal
        rowCount += 1
    treeTestCount += 1

for i in range(testArrLen):
    for j in range(500):
        testPrediction[i][j] = "1" if testPredictionList[i][j] >= 0 else "0"

for i in range(500):
    errorCount = 0
    for j in range(testArrLen):
        if testArr[j][lastIndx] != testPrediction[j][i]:
            errorCount += 1
    test_errors.append(errorCount/testArrLen)

plt.figure(figsize=(12, 5))
plt.plot(range(1, 500 + 1), train_errors, label='Train Error')
plt.plot(range(1, 500 + 1), test_errors, label='Test Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.title('Train and Test Errors vs. Iterations')
plt.legend()
plt.tight_layout()
plt.show()
     
#Driver Function -------------------------- End
