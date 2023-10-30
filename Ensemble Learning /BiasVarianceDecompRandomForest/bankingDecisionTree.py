import math
import random
import numpy as np
import matplotlib.pyplot as plt

#Common Code --------------------------------------------- Start

featAttrDict = {
    "age": {
        "yes": 0,
        "no": 0,
    },
    "job": {
        "admin.": 0,
        "unknown": 0,
        "unemployed": 0,
        "management": 0,
        "housemaid": 0,
        "entrepreneur": 0,
        "student": 0,
        "blue-collar": 0,
        "self-employed": 0,
        "retired": 0,
        "technician": 0,
        "services": 0
    },
    "marital": {
        "married": 0,
        "divorced": 0,
        "single": 0
    },
    "education": {
        "unknown": 0,
        "secondary": 0,
        "primary": 0,
        "tertiary": 0
    },
    "default": {
        "yes": 0,
        "no": 0
    },
    "balance": {
        "yes": 0,
        "no": 0
    },
    "housing": {
        "yes": 0,
        "no": 0
    },
    "loan": {
        "yes": 0,
        "no": 0
    },
    "contact": {
        "unknown": 0,
        "telephone": 0,
        "cellular": 0
    },
    "day": {
        "yes": 0,
        "no": 0
    },
    "month": {
        "jan": 0, 
        "feb": 0, 
        "mar": 0,
        "apr": 0,
        "may": 0, 
        "jun": 0, 
        "jul": 0,
        "aug": 0,
        "sep": 0,
        "oct": 0,
        "nov": 0,
        "dec": 0
    },
    "duration": {
        "yes": 0,
        "no": 0
    },
    "campaign": {
        "yes": 0,
        "no": 0
    },
    "pdays": {
        "yes": 0,
        "no": 0
    },
    "previous": {
        "yes": 0,
        "no": 0
    },
    "poutcome": {
        "unknown": 0,
        "other": 0,
        "failure": 0,
        "success": 0
    },
    "y": {
        "yes": 0,
        "no": 0
    }
}

dataList = []
with open('train.csv', 'r') as f:
    i = 0
    for line in f:
        terms = line.strip().split(',')
        dataList.append(terms)
        i += 1
arr = np.array(dataList)

testDataList = []
with open('test.csv', 'r') as f:
    i = 0
    for line in f:
        terms = line.strip().split(',')
        testDataList.append(terms)
        i += 1
testArr = np.array(testDataList)

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
        #age
        if (int(line[0]) > 25) and (int(line[0]) < 50):
            line[0] = "yes"
        else:
            line[0] = "no" 
        #balance
        if int(line[5]) > 500:
            line[5] = "yes"
        else:
            line[5] = "no"
        #day
        if int(line[9]) > 15:
            line[9] = "yes"
        else:
            line[9] = "no"
        #duration
        if int(line[11]) > 180:
            line[11] = "no"
        else:
            line[11] = "yes"
        #campaign
        if int(line[12]) > 2:
            line[12] = "no"
        else:
            line[12] = "yes"
        #pdays
        if (int(line[13]) == -1) or (int(line[13]) < 200):
            line[13] = "no"
        else:
            line[13] = "yes"
        #previous
        if int(line[14]) > 2:
            line[14] = "yes"
        else:
            line[14] = "no"


def setContiniousValueTest():
    for line in testArr:
        #age
        if (int(line[0]) > 25) and (int(line[0]) < 50):
            line[0] = "yes"
        else:
            line[0] = "no" 
        #balance
        if int(line[5]) > 500:
            line[5] = "yes"
        else:
            line[5] = "no"
        #day
        if int(line[9]) > 15:
            line[9] = "yes"
        else:
            line[9] = "no"
        #duration
        if int(line[11]) > 180:
            line[11] = "no"
        else:
            line[11] = "yes"
        #campaign
        if int(line[12]) > 2:
            line[12] = "no"
        else:
            line[12] = "yes"
        #pdays
        if (int(line[13]) == -1) or (int(line[13]) < 200):
            line[13] = "no"
        else:
            line[13] = "yes"
        #previous
        if int(line[14]) > 2:
            line[14] = "yes"
        else:
            line[14] = "no"

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
        "yes": 0,
        "no": 0,
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    if count > 0:
        prob1 = (attrLabelDict["yes"]/count) if (attrLabelDict["yes"]/count) > 0.0 else 1
        prob2 = (attrLabelDict["no"]/count) if (attrLabelDict["no"]/count) > 0.0 else 1
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


def TestData(rootNode):
    testdataList = []

    #change the file name to test.csv or train.csv to run tests with test or training data respectively
    with open('test.csv', 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            testdataList.append(terms)
    
    testArr = np.array(testdataList)

    testContArr = setContiniousValueTest(testArr)

    totalCount = len(testContArr)
    passCount = 0
    with open("myfile.txt", "w") as w:
        for line in testContArr:
            if TestPredict(line, rootNode, np.array(["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"])) == line[16]:
                w.write("Pass \n")
                passCount += 1
            else:
                w.write("Fail \n")
        fail = (totalCount - passCount)/totalCount
        w.write(f'Prediction Error: {fail}')
    
#Testing - Traversing the tree ---------------------------- Start            

def GetPrediction(rootNode, line):
    return TestPredict(line, rootNode, np.array(["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]))


#Driver Function -------------------------- Start

def SampleBiasVariance(predictorsList):
    colCount = len(testArr[0]) - 1
    biasList = []
    varianceList = []
    for line in testArr:
        predictionList = []
        sum = 0
        for predictor in predictorsList:
            prediction = 1 if GetPrediction(predictor, line) == "yes" else -1 
            predictionList.append(prediction)
            sum += prediction
        average = sum/100
        bias = (average - (1 if line[colCount] == "yes" else -1)) ** 2
        predictionList = np.array(predictionList)
        variance = np.sum(((predictionList - average) ** 2)) / 99
        biasList.append(bias)
        varianceList.append(variance)
    
    biasAvg = np.sum(biasList) / len(testArr)
    varianceAvg = np.sum(varianceList) / len(testArr)
    biasVarianceAvg = biasAvg + varianceAvg
    return biasAvg, varianceAvg, biasVarianceAvg


def BiasVariance(randomForestPredictors):
    colCount = len(testArr[0]) - 1
    biasList = []
    varianceList = []
    for line in testArr:
        predictionList = []
        sum = 0
        for treeList in randomForestPredictors:
            baggedSum = 0
            for node in treeList:
                baggedSum += 1 if GetPrediction(node, line) == "yes" else -1
            prediction = 1 if baggedSum >= 0 else -1
            predictionList.append(prediction)
            sum += prediction
        average = sum / 100
        bias = (average - (1 if line[colCount] == "yes" else -1)) ** 2
        predictionList = np.array(predictionList)
        variance = np.sum(((predictionList - average) ** 2)) / 99
        biasList.append(bias)
        varianceList.append(variance)
    
    biasAvg = np.sum(biasList) / len(testArr)
    varianceAvg = np.sum(varianceList) / len(testArr)
    biasVarianceAvg = biasAvg + varianceAvg
    return biasAvg, varianceAvg, biasVarianceAvg


def DecisionDriver():
    setContiniousValue()
    setContiniousValueTest()

    maxDepth = 20
    T = 500
    samplePick = 1000

    randomForestPredictors = []
    firstPredictors = []
    for itr in range(100):
        treeList = []
        for i in range(T):
            newArr = np.array(random.choices(arr, k = samplePick))
            node = ID3RandTreeLearn(newArr, np.array(["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]), 0, maxDepth)
            treeList.append(node)

        firstPredictors.append(treeList[0])
        randomForestPredictors.append(treeList)

    print("Single Tree Predictor Bias, Variance, Squared error:", SampleBiasVariance(firstPredictors))
    print("Whole Forest Predictor Bias, Variance, Squared error", BiasVariance(randomForestPredictors))

DecisionDriver()

#Driver Function -------------------------- End

#Result
# Single Tree Predictor Bias, Variance, Squared error: (0.33520248, 0.36247830303030304, 0.6976807830303031)
# Whole Forest Predictor Bias, Variance, Squared error (0.41825528000000006, 0.015491636363636365, 0.43374691636363644)