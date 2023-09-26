import math
import numpy as np

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


def setContiniousValueTest(testData):
    for line in testData:
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
    
    return testData


def FixTrainingData():
    for key in featAttrDict:
        if "unknown" in featAttrDict[key]:
            del featAttrDict[key]["unknown"]

    majorityDict = {
        0: max(featAttrDict["age"], key = featAttrDict["age"].get),
        1: max(featAttrDict["job"], key = featAttrDict["job"].get),
        2: max(featAttrDict["marital"], key = featAttrDict["marital"].get),
        3: max(featAttrDict["education"], key = featAttrDict["education"].get),
        4: max(featAttrDict["default"], key = featAttrDict["default"].get),
        5: max(featAttrDict["balance"], key = featAttrDict["balance"].get),
        6: max(featAttrDict["housing"], key = featAttrDict["housing"].get),
        7: max(featAttrDict["loan"], key = featAttrDict["loan"].get),
        8: max(featAttrDict["contact"], key = featAttrDict["contact"].get),
        9: max(featAttrDict["day"], key = featAttrDict["day"].get),
        10: max(featAttrDict["month"], key = featAttrDict["month"].get),
        11: max(featAttrDict["duration"], key = featAttrDict["duration"].get),
        12: max(featAttrDict["campaign"], key = featAttrDict["campaign"].get),
        13: max(featAttrDict["pdays"], key = featAttrDict["pdays"].get),
        14: max(featAttrDict["previous"], key = featAttrDict["previous"].get),
        15: max(featAttrDict["poutcome"], key = featAttrDict["poutcome"].get),
        16: max(featAttrDict["y"], key = featAttrDict["y"].get)
    }
    for line in arr:
        i = 0
        for cell in line:
            if cell == "unknown":
                line[i] = majorityDict[i]
            i += 1


def FixTestData(testData):
    for key in featAttrDict:
        if "unknown" in featAttrDict[key]:
            del featAttrDict[key]["unknown"]

    majorityDict = {
        0: max(featAttrDict["age"], key = featAttrDict["age"].get),
        1: max(featAttrDict["job"], key = featAttrDict["job"].get),
        2: max(featAttrDict["marital"], key = featAttrDict["marital"].get),
        3: max(featAttrDict["education"], key = featAttrDict["education"].get),
        4: max(featAttrDict["default"], key = featAttrDict["default"].get),
        5: max(featAttrDict["balance"], key = featAttrDict["balance"].get),
        6: max(featAttrDict["housing"], key = featAttrDict["housing"].get),
        7: max(featAttrDict["loan"], key = featAttrDict["loan"].get),
        8: max(featAttrDict["contact"], key = featAttrDict["contact"].get),
        9: max(featAttrDict["day"], key = featAttrDict["day"].get),
        10: max(featAttrDict["month"], key = featAttrDict["month"].get),
        11: max(featAttrDict["duration"], key = featAttrDict["duration"].get),
        12: max(featAttrDict["campaign"], key = featAttrDict["campaign"].get),
        13: max(featAttrDict["pdays"], key = featAttrDict["pdays"].get),
        14: max(featAttrDict["previous"], key = featAttrDict["previous"].get),
        15: max(featAttrDict["poutcome"], key = featAttrDict["poutcome"].get),
        16: max(featAttrDict["y"], key = featAttrDict["y"].get)
    }
    for line in testData:
        i = 0
        for cell in line:
            if cell == "unknown":
                line[i] = majorityDict[i]
            i += 1

    return testData

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
    minLabelCount = featAttrDict["y"]["yes"]
    mError = 0

    for label in featAttrDict["y"]:
        if featAttrDict["y"][label] > 0:
            labelCount += 1

    if labelCount > 1:
        for label in featAttrDict["y"]:
            if featAttrDict["y"][label] < minLabelCount:
                minLabelCount = featAttrDict["y"][label]
        mError = minLabelCount / totalCount
    return mError


def attributeMajority(dataset, attribute):
    majorityError = 0
    attrLabelDict = {
        "yes": 0,
        "no": 0
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    
    if count > 0:
        minAttrVal = min(attrLabelDict.values()) * 1.0
        majorityError = minAttrVal/count

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

    for label in featAttrDict["y"]:
        if featAttrDict["y"][label] > 0:
            labelCount += 1

    if labelCount > 1:
        total = 0
        for label in featAttrDict["y"]:
            probSquare = (featAttrDict["y"][label]/(totalCount * 1.0)) ** 2
            total += probSquare
        giniIndex = 1 - total

    return giniIndex


def attributeGini(dataset, attribute):
    dataLen = len(dataset) * 1.0
    gini = 0
    attrLabelDict = {
        "yes": 0,
        "no": 0
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    if count > 0:
        prob1 = (attrLabelDict["yes"]/count) ** 2
        prob2 = (attrLabelDict["no"]/count) ** 2
        probSum = prob1 + prob2
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
        return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
    
    colWithLowestEntropy = findLowestEntropyFeature(dataset, colNames)
    colPos = np.where(colNames == colWithLowestEntropy)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestEntropy]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
        trimDataset = np.delete(trimDataset, colPos, 1)
        children[branch] = ID3DecisionEntropyTree(trimDataset, colNames, depth+1, maxDepth)

    return Node(colWithLowestEntropy, depth, False, children)


def ID3DecisionMajorityTree(dataset, colNames, depth, maxDepth):
    setFeatureAttrCount(dataset, colNames)
    H = overallMajority(dataset)

    if depth == maxDepth or H == 0:
        return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
    
    colWithLowestME = findLowestMajorityFeature(dataset, colNames)
    colPos = np.where(colNames == colWithLowestME)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestME]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
        trimDataset = np.delete(trimDataset, colPos, 1)
        children[branch] = ID3DecisionMajorityTree(trimDataset, colNames, depth+1, maxDepth)

    return Node(colWithLowestME, depth, False, children)


def ID3DecisionGiniTree(dataset, colNames, depth, maxDepth):
    setFeatureAttrCount(dataset, colNames)
    H = overallGini(dataset)
    if depth == maxDepth or H == 0:
        return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
    
    colWithLowestGini = findLowestGiniFeature(dataset, colNames)
    colPos = np.where(colNames == colWithLowestGini)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestGini]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
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

    with open('test.csv', 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            testdataList.append(terms)
    
    testArr = np.array(testdataList)

    testContArr = setContiniousValueTest(testArr)
    testData = FixTestData(testContArr)
    totalCount = len(testData)
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


#Driver Function -------------------------- Start

def DecisionDriver():
    setContiniousValue()
    setFeatureAttrCount(arr, ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
    FixTrainingData()
    setFeatureAttrCount(arr, ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
    maxDepth = 16
    node = ID3DecisionMajorityTree(arr, np.array(["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]), 0, maxDepth)
    
    TestData(node)

DecisionDriver()

#Driver Function -------------------------- End