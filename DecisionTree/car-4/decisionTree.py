import math
import numpy as np

#Common Code --------------------------------------------- Start

featAttrDict = {
    "buying": {
        "vhigh": 0,
        "high": 0,
        "med": 0,
        "low": 0
    },
    "maint": {
        "vhigh": 0,
        "high": 0,
        "med": 0,
        "low": 0
    },
    "doors": {
        "2": 0,
        "3": 0,
        "4": 0,
        "5more": 0
    },
    "persons": {
        "2": 0,
        "4": 0,
        "more": 0
    },
    "lug_boot": {
        "small": 0,
        "med": 0,
        "big": 0
    },
    "safety": {
        "low": 0,
        "med": 0,
        "high": 0
    },
    "label": {
        "unacc": 0,
        "acc": 0,
        "good": 0,
        "vgood": 0
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

def FixTrainingData():
    majorityDict = {
        0: max(featAttrDict["buying"], key = featAttrDict["buying"].get),
        1: max(featAttrDict["maint"], key = featAttrDict["maint"].get),
        2: max(featAttrDict["doors"], key = featAttrDict["doors"].get),
        3: max(featAttrDict["persons"], key = featAttrDict["persons"].get),
        4: max(featAttrDict["lug_boot"], key = featAttrDict["lug_boot"].get),
        5: max(featAttrDict["safety"], key = featAttrDict["safety"].get),
        6: max(featAttrDict["label"], key = featAttrDict["label"].get),
    }

    for line in arr:
        i = 0
        for cell in line:
            if not cell:
                line[i] = majorityDict[i]
            i += 1

def FixTestData(testData):
    majorityDict = {
        0: max(featAttrDict["buying"], key = featAttrDict["buying"].get),
        1: max(featAttrDict["maint"], key = featAttrDict["maint"].get),
        2: max(featAttrDict["doors"], key = featAttrDict["doors"].get),
        3: max(featAttrDict["persons"], key = featAttrDict["persons"].get),
        4: max(featAttrDict["lug_boot"], key = featAttrDict["lug_boot"].get),
        5: max(featAttrDict["safety"], key = featAttrDict["safety"].get),
        6: max(featAttrDict["label"], key = featAttrDict["label"].get),
    }

    for line in testData:
        i = 0
        for cell in line:
            if not cell:
                line[i] = majorityDict[i]
            i += 1

    return testData

#Common Code --------------------------------------------- End


#Entropy Calculation -------------------------------------- Start

def overallEntropy(dataset):
    totalCount = len(dataset)
    labelCount = 0
    entropy = 0

    for label in featAttrDict["label"]:
        if featAttrDict["label"][label] > 0:
            labelCount += 1

    if labelCount > 1:
        for label in featAttrDict["label"]:
            calc = featAttrDict["label"][label]/(totalCount * 1.0)
            prob = calc if calc > 0.0 else 1
            entropy += (prob) * math.log(prob, 2)
    return -entropy   


def attributeEntropy(dataset, attribute):
    dataLen = len(dataset) * 1.0
    entropy = 0
    attrLabelDict = {
        "unacc": 0,
        "acc": 0,
        "good": 0,
        "vgood": 0
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    if count > 0:
        prob1 = (attrLabelDict["unacc"]/count) if (attrLabelDict["unacc"]/count) > 0.0 else 1
        prob2 = (attrLabelDict["acc"]/count) if (attrLabelDict["acc"]/count) > 0.0 else 1
        prob3 = (attrLabelDict["good"]/count) if (attrLabelDict["good"]/count) > 0.0 else 1
        prob4 = (attrLabelDict["vgood"]/count) if (attrLabelDict["vgood"]/count) > 0.0 else 1
        entropy = - prob1 * math.log(prob1, 2) - prob2 * math.log(prob2, 2) - prob3 * math.log(prob3, 2) - prob4 * math.log(prob4, 2)

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
    minEntropyCol = colNameArr[0]
    totalCol = len(dataset[0])
    colCount = 0
    for feature in colNameArr[:-1]:
        featureLabelCol = dataset[:, [colCount, totalCol - 1]]
        entropy = featureEntropy(featureLabelCol, feature)
        if entropy < minEntropy:
            minEntropy = entropy
            minEntropyCol = feature
        colCount += 1
    return minEntropyCol

#Entropy Calculation -------------------------------------- End

#Majority Error Calculation ------------------------------- Start

def overallMajority(dataset):
    totalCount = len(dataset) * 1.0
    labelCount = 0
    minLabelCount = featAttrDict["label"]["unacc"]
    mError = 0

    for label in featAttrDict["label"]:
        if featAttrDict["label"][label] > 0:
            labelCount += 1

    if labelCount > 1:
        for label in featAttrDict["label"]:
            if featAttrDict["label"][label] < minLabelCount:
                minLabelCount = featAttrDict["label"][label]
        mError = minLabelCount / totalCount
    return mError


def attributeMajority(dataset, attribute):
    dataLen = len(dataset) * 1.0
    majorityError = 0
    attrLabelDict = {
        "unacc": 0,
        "acc": 0,
        "good": 0,
        "vgood": 0
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    
    if count > 0:
        minAttrVal = min(attrLabelDict.values()) * 1.0
        majorityError = minAttrVal/count
    #     for attr in attrLabelDict:

    #     prob1 = (attrLabelDict["unacc"]/count)
    #     prob2 = (attrLabelDict["acc"]/count)
    #     prob3 = (attrLabelDict["good"]/count)
    #     prob4 = (attrLabelDict["vgood"]/count)
        # majorityError += - prob1 * math.log(prob1, 2) - prob2 * math.log(prob2, 2) - prob3 * math.log(prob3, 2) - prob4 * math.log(prob4, 2)

    return majorityError


def featureMajority(data, featureName):
    majorityError = 0
    totalSize = len(data)*1.0
    for attribute in featAttrDict[featureName]:
        if featAttrDict[featureName][attribute] > 0:
            majorityError += (featAttrDict[featureName][attribute]/totalSize) * attributeMajority(data, attribute)
    return majorityError


def findLowestMajorityFeature(dataset, colNameArr):
    minMajorityError = 2
    minMajorityErrorCol = colNameArr[0]
    totalCol = len(dataset[0])
    colCount = 0
    for feature in colNameArr[:-1]:
        featureLabelCol = dataset[:, [colCount, totalCol - 1]]
        majorityError = featureMajority(featureLabelCol, feature)
        if majorityError < minMajorityError:
            minMajorityError = majorityError
            minMajorityErrorCol = feature
        colCount += 1
    return minMajorityErrorCol

#Majority Error Calculation ------------------------------- End

#Gini Index Calculation ------------------------------- Start

def overallGini(dataset):
    totalCount = len(dataset)
    labelCount = 0
    giniIndex = 0

    for label in featAttrDict["label"]:
        if featAttrDict["label"][label] > 0:
            labelCount += 1

    if labelCount > 1:
        total = 0
        for label in featAttrDict["label"]:
            probSquare = (featAttrDict["label"][label]/(totalCount * 1.0)) ** 2
            total += probSquare
        giniIndex = 1 - total

    return giniIndex


def attributeGini(dataset, attribute):
    dataLen = len(dataset) * 1.0
    gini = 0
    attrLabelDict = {
        "unacc": 0,
        "acc": 0,
        "good": 0,
        "vgood": 0
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    if count > 0:
        prob1 = (attrLabelDict["unacc"]/count) ** 2
        prob2 = (attrLabelDict["acc"]/count) ** 2
        prob3 = (attrLabelDict["good"]/count) ** 2
        prob4 = (attrLabelDict["vgood"]/count) ** 2
        probSum = prob1 + prob2 + prob3 + prob4
        gini = 1 - probSum

    return gini


def featureGini(data, featureName):
    gini = 0
    totalSize = len(data)*1.0
    for attribute in featAttrDict[featureName]:
        if featAttrDict[featureName][attribute] > 0:
            gini += (featAttrDict[featureName][attribute]/totalSize) * attributeGini(data, attribute)
    return gini


def findLowestGiniFeature(dataset, colNameArr):
    minGini = 2
    minGiniCol = colNameArr[0]
    totalCol = len(dataset[0])
    colCount = 0
    for feature in colNameArr[:-1]:
        featureLabelCol = dataset[:, [colCount, totalCol - 1]]
        gini = featureGini(featureLabelCol, feature)
        if gini < minGini:
            minGini = gini
            minGiniCol = feature
        colCount += 1
    return minGiniCol

#Gini Index Calculation ------------------------------- End

#ID3 Recursion ------------------------------------------- Start

def ID3DecisionEntropyTree(dataset, colNames, depth, maxDepth):
    setFeatureAttrCount(dataset, colNames)
    H = overallEntropy(dataset)
    if depth == maxDepth or H == 0:
        return Node("label", depth, True, max(featAttrDict["label"], key=featAttrDict["label"].get))
    
    colWithLowestEntropy = findLowestEntropyFeature(dataset, colNames)
    colPos = np.where(colNames == colWithLowestEntropy)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestEntropy]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            return Node("label", depth, True, max(featAttrDict["label"], key=featAttrDict["label"].get))
        trimDataset = np.delete(trimDataset, colPos, 1)
        children[branch] = ID3DecisionEntropyTree(trimDataset, colNames, depth+1, maxDepth)

    return Node(colWithLowestEntropy, depth, False, children)


def ID3DecisionMajorityTree(dataset, colNames, depth, maxDepth):
    setFeatureAttrCount(dataset, colNames)
    H = overallMajority(dataset)

    if depth == maxDepth or H == 0:
        return Node("label", depth, True, max(featAttrDict["label"], key=featAttrDict["label"].get))
    
    colWithLowestME = findLowestMajorityFeature(dataset, colNames)
    colPos = np.where(colNames == colWithLowestME)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestME]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            return Node("label", depth, True, max(featAttrDict["label"], key=featAttrDict["label"].get))
        trimDataset = np.delete(trimDataset, colPos, 1)
        children[branch] = ID3DecisionMajorityTree(trimDataset, colNames, depth+1, maxDepth)

    return Node(colWithLowestME, depth, False, children)


def ID3DecisionGiniTree(dataset, colNames, depth, maxDepth):
    setFeatureAttrCount(dataset, colNames)
    H = overallGini(dataset)
    if depth == maxDepth or H == 0:
        return Node("label", depth, True, max(featAttrDict["label"], key=featAttrDict["label"].get))
    
    colWithLowestGini = findLowestGiniFeature(dataset, colNames)
    colPos = np.where(colNames == colWithLowestGini)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestGini]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            return Node("label", depth, True, max(featAttrDict["label"], key=featAttrDict["label"].get))
        trimDataset = np.delete(trimDataset, colPos, 1)
        children[branch] = ID3DecisionGiniTree(trimDataset, colNames, depth+1, maxDepth)

    return Node(colWithLowestGini, depth, False, children)

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

    with open('train.csv', 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            testdataList.append(terms)
    
    testArr = np.array(testdataList)

    # setFeatureAttrCount(testArr, ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])
    testData = FixTestData(testArr)
    totalCount = len(testdataList)
    passCount = 0
    with open("myfile.txt", "w") as w:
        for line in testData:
            if TestPredict(line, rootNode, np.array(["buying", "maint", "doors", "persons", "lug_boot", "safety"])) == line[6]:
                w.write("Pass \n")
                passCount += 1
            else:
                w.write("Fail \n")
        failPercent = (totalCount - passCount)/totalCount
        w.write(f'Prediction Error: {failPercent}')
    
#Testing - Traversing the tree ---------------------------- Start            


#Driver Function -------------------------- Start

def DecisionDriver():
    setFeatureAttrCount(arr, ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])
    FixTrainingData()
    setFeatureAttrCount(arr, ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])
    maxDepth = 6
    node = ID3DecisionEntropyTree(arr, np.array(["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]), 0, maxDepth)
    
    TestData(node)

DecisionDriver()

#Driver Function -------------------------- End