import math
import random
import numpy as np
import matplotlib.pyplot as plt

#Common Code --------------------------------------------- Start

featAttrDict = {
    "age": {
        "veryLow": 0, # < 25
        "low": 0,  # < 40
        "med": 0,  # < 60
        "high": 0,  # rest
        "?": 0
    },
    #if ? assign private
    "workclass": {
        "Private": 0, 
        "Self-emp-not-inc": 0, 
        "Self-emp-inc": 0, 
        "Federal-gov": 0, 
        "Local-gov": 0, 
        "State-gov": 0, 
        "Without-pay": 0, 
        "Never-worked": 0,
        "?": 0
    },
    #if ? assign med
    "fnlwgt": {
        "low": 0, # < 100000 
        "med": 0, # < 350000
        "high": 0, # Rest
        "?": 0
    },
    "education": {
        "Bachelors": 0, 
        "Some-college": 0, 
        "11th": 0, 
        "HS-grad": 0, 
        "Prof-school": 0, 
        "Assoc-acdm": 0, 
        "Assoc-voc": 0, 
        "9th": 0, 
        "7th-8th": 0, 
        "12th": 0, 
        "Masters": 0, 
        "1st-4th": 0, 
        "10th": 0, 
        "Doctorate": 0, 
        "5th-6th": 0, 
        "Preschool": 0,
        "?": 0
    },
    #if ? assign med
    "education-num": {
        "low": 0,  # < 8
        "med": 0,  # , 12
        "high": 0,  #rest
        "?": 0
    },
    "marital-status": {
        "Married-civ-spouse": 0, 
        "Divorced": 0, 
        "Never-married": 0, 
        "Separated": 0, 
        "Widowed": 0, 
        "Married-spouse-absent": 0, 
        "Married-AF-spouse": 0,
        "?": 0
    },
    "occupation": {
        "Tech-support": 0, 
        "Craft-repair": 0, 
        "Other-service": 0, 
        "Sales": 0, 
        "Exec-managerial": 0, 
        "Prof-specialty": 0, 
        "Handlers-cleaners": 0, 
        "Machine-op-inspct": 0, 
        "Adm-clerical": 0, 
        "Farming-fishing": 0, 
        "Transport-moving": 0, 
        "Priv-house-serv": 0, 
        "Protective-serv": 0, 
        "Armed-Forces": 0,
        "?": 0
    },
    "relationship": {
        "Wife": 0, 
        "Own-child": 0, 
        "Husband": 0, 
        "Not-in-family": 0, 
        "Other-relative": 0, 
        "Unmarried": 0,
        "?": 0
    },
    #if ? assign white
    "race": {
        "White": 0, 
        "Asian-Pac-Islander": 0, 
        "Amer-Indian-Eskimo": 0, 
        "Other": 0, 
        "Black": 0,
        "?": 0
    },
    "sex": {
        "Female": 0,
        "Male": 0,
        "?": 0
    },
    #if ? assign low
    "capital-gain": {
        "low": 0, # < 1000
        "med": 0, # < 10000
        "high": 0, # rest
        "?": 0
    },
    #if ? assign low
    "capital-loss": {
        "low": 0, # < 1000
        "high": 0, # rest
        "?": 0
    },
    #if ? assign med
    "hours-per-week": {
        "low": 0, # < 25
        "med": 0, # < 45
        "high": 0, # rest
        "?": 0
    },
    #if ? assign US
    "native-country": {
        "United-States": 0, 
        "Cambodia": 0, 
        "England": 0, 
        "Puerto-Rico": 0, 
        "Canada": 0, 
        "Germany": 0, 
        "Outlying-US(Guam-USVI-etc)": 0, 
        "India": 0, 
        "Japan": 0, 
        "Greece": 0, 
        "South": 0, 
        "China": 0, 
        "Cuba": 0, 
        "Iran": 0, 
        "Honduras": 0, 
        "Philippines": 0, 
        "Italy": 0, 
        "Poland": 0, 
        "Jamaica": 0, 
        "Vietnam": 0, 
        "Mexico": 0, 
        "Portugal": 0, 
        "Ireland": 0, 
        "France": 0, 
        "Dominican-Republic": 0, 
        "Laos": 0, 
        "Ecuador": 0, 
        "Taiwan": 0, 
        "Haiti": 0, 
        "Columbia": 0, 
        "Hungary": 0, 
        "Guatemala": 0, 
        "Nicaragua": 0, 
        "Scotland": 0, 
        "Thailand": 0, 
        "Yugoslavia": 0, 
        "El-Salvador": 0, 
        "Trinadad&Tobago": 0, 
        "Peru": 0,
        "Hong": 0, 
        "Holand-Netherlands": 0,
        "?": 0
    },
    "y": {
        "1": 0,
        "0": 0
    }
}

dataList = []
with open('train_final.csv', 'r') as f:
    next(f)
    i = 0
    for line in f:
        terms = line.strip().split(',')
        dataList.append(terms)
        i += 1
arr = np.array(dataList)

testDataList = []
with open('test_final.csv', 'r') as f:
    next(f)
    i = 0
    for line in f:
        terms = line.strip().split(',')
        testDataList.append(terms)
        i += 1
testArr = np.array(testDataList)[:,1:]

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
        line0 = int(line[0])
        if line0 < 25:
            line[0] = "veryLow"
        elif line0 < 40:
            line[0] = "low"
        elif line0 < 60:
            line[0] = "med" 
        else:
            line[0] = "high" 

        #fnlwgt
        line2 = int(line[2])
        if line2 < 100000:
            line[2] = "low"
        elif line2 < 350000:
            line[2] = "med"
        else:
            line[2] = "high"

        #education-num
        line4 = int(line[4])
        if line4 < 8:
            line[4] = "low"
        elif line4 < 12:
            line[4] = "med"
        else:
            line[4] = "high"

        #capital-gain
        line10 = int(line[10])
        if line10 < 1000:
            line[10] = "low"
        elif line10 < 10000:
            line[10] = "med"
        else:
            line[10] = "high"

        #capital-loss
        line11 = int(line[11])
        if line11 < 1000:
            line[11] = "low"
        else:
            line[11] = "high"

        #hours-per-week
        line12 = int(line[12])
        if line12 < 25:
            line[12] = "low"
        elif line12 < 45:
            line[12] = "med"
        else:
            line[12] = "high"


def setContiniousValueTest():
    for line in testArr:
        #age
        line0 = int(line[0])
        if line0 < 25:
            line[0] = "veryLow"
        elif line0 < 40:
            line[0] = "low"
        elif line0 < 60:
            line[0] = "med" 
        else:
            line[0] = "high" 

        #fnlwgt
        line2 = int(line[2])
        if line2 < 100000:
            line[2] = "low"
        elif line2 < 350000:
            line[2] = "med"
        else:
            line[2] = "high"

        #education-num
        line4 = int(line[4])
        if line4 < 8:
            line[4] = "low"
        elif line4 < 12:
            line[4] = "med"
        else:
            line[4] = "high"

        #capital-gain
        line10 = int(line[10])
        if line10 < 1000:
            line[10] = "low"
        elif line10 < 10000:
            line[10] = "med"
        else:
            line[10] = "high"

        #capital-loss
        line11 = int(line[11])
        if line11 < 1000:
            line[11] = "low"
        else:
            line[11] = "high"

        #hours-per-week
        line12 = int(line[12])
        if line12 < 25:
            line[12] = "low"
        elif line12 < 45:
            line[12] = "med"
        else:
            line[12] = "high"

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
        "1": 0,
        "0": 0,
    }
    count = 0
    for line in dataset:
        if line[0] == attribute:
            count += 1
            attrLabelDict[line[1]] += 1

    if count > 0:
        prob1 = (attrLabelDict["1"]/count) if (attrLabelDict["1"]/count) > 0.0 else 1
        prob2 = (attrLabelDict["0"]/count) if (attrLabelDict["0"]/count) > 0.0 else 1
        entropy = - prob1 * math.log(prob1, 2) - prob2 * math.log(prob2, 2)

    return entropy


def featureEntropy(data, featureName):
    entropy = 0
    totalSize = len(data)*1.0
    for attribute in featAttrDict[featureName]:
        if featAttrDict[featureName][attribute] > 0:
            entropy += (featAttrDict[featureName][attribute]/totalSize) * attributeEntropy(data, attribute)
    return entropy


def findLowestEntropyFeature(dataset, colNameArr, features):
    minEntropy = 2
    colLen = len(colNameArr) - 1
    pickCount = features if features < colLen else colLen
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

def ID3RandTreeLearn(dataset, colNames, depth, maxDepth, featureCount):
    setFeatureAttrCount(dataset, colNames)
    H = overallEntropy(dataset)
    if H == 0:
        return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
    
    colWithLowestEntropy = findLowestEntropyFeature(dataset, colNames, featureCount)
    colPos = np.where(colNames == colWithLowestEntropy)[0][0]
    colNames = np.delete(colNames, colPos)
    children = {}
    for branch in featAttrDict[colWithLowestEntropy]:
        trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
        if len(trimDataset) == 0:
            children[branch] = Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
        else:
            trimDataset = np.delete(trimDataset, colPos, 1)
            children[branch] = ID3RandTreeLearn(trimDataset, colNames, depth+1, maxDepth, featureCount)

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
    
#Testing - Traversing the tree ---------------------------- Start            

def GetPrediction(rootNode, line):
    return TestPredict(line, rootNode, np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]))

treeList = []
#Driver Function -------------------------- Start

def DecisionDriver(features):
    #Change the max depth value here to change the depth of the decision tree.
    maxDepth = 20
    T = 500
    samplePick = 1000
    # train_error_List = []
    # test_error_List = []
    # dataLen = len(arr)
    # testDataLen = len(testArr)

    for i in range(T):
        newArr = np.array(random.choices(arr, k = samplePick))
        node = ID3RandTreeLearn(newArr, np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "y"]), 0, maxDepth, features)
        treeList.append(node)

        # last = len(arr[0]) - 1
        # predictResults = []
        

        # for line in arr:
        #     sum = 0
        #     for node in treeList:
        #         sum += 1 if GetPrediction(node, line) == "1" else -1
        #     res = "1" if sum >= 0 else "0"
        #     predictResults.append(res)

    testPredictResults = []
    for line in testArr:
        sum = 0
        for node in treeList:
            sum += 1 if GetPrediction(node, line) == "1" else -1
        res = "1" if sum >= 0 else "0"
        testPredictResults.append(res)

    with open("myfile.csv", "w") as w:
        w.write("ID,Prediction\n")
        id = 1
        for prediction in testPredictResults:
            w.write(f'{id},{prediction}\n')
            id += 1

        # count = 0
        # errorCount = 0
        # for line in arr:
        #     if predictResults[count] != line[last]:
        #         errorCount += 1
        #     count += 1

        # testCount = 0
        # testErrorCount = 0
        # for line in testArr:
        #     if testPredictResults[testCount] != line[last]:
        #         testErrorCount += 1
        #     testCount += 1

        # train_error_List.append(errorCount/dataLen)
        # test_error_List.append(testErrorCount/testDataLen)

    # return test_error_List
    
setContiniousValue()
setContiniousValueTest()
# train_errors_2, test_errors_2 = DecisionDriver(2)
# train_errors_4, test_errors_4 = DecisionDriver(4)
DecisionDriver(6)

# plt.figure(figsize=(12, 5))
# plt.plot(range(1, 500 + 1), train_errors_2, label='Training Error')
# plt.plot(range(1, 500 + 1), test_errors_2, label='Test Error')
# plt.xlabel('Number of Iterations (T)')
# plt.ylabel('Error')
# plt.title('Training and Test Errors vs. Iterations For 2 Features')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 5))
# plt.plot(range(1, 500 + 1), train_errors_4, label='Training Error')
# plt.plot(range(1, 500 + 1), test_errors_4, label='Test Error')
# plt.xlabel('Number of Iterations (T)')
# plt.ylabel('Error')
# plt.title('Training and Test Errors vs. Iterations 4 Features')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 5))
# plt.plot(range(1, 500 + 1), train_errors_6, label='Training Error')
# plt.plot(range(1, 500 + 1), test_errors_6, label='Test Error')
# plt.xlabel('Number of Iterations (T)')
# plt.ylabel('Error')
# plt.title('Training and Test Errors vs. Iterations 6 Features')
# plt.legend()
# plt.tight_layout()
# plt.show()

#Driver Function -------------------------- End