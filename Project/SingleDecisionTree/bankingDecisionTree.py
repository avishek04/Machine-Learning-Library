import math
import numpy as np

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
        featAttrDict[feature]["?"] = 0
    del featAttrDict["y"]["?"]



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


def setContiniousValueTest(testData):
    for line in testData:
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
    
    return testData


def FixTrainingData():
    for key in featAttrDict:
        if "?" in featAttrDict[key]:
            del featAttrDict[key]["?"]

    majorityDict = {
        0: max(featAttrDict["age"], key = featAttrDict["age"].get),
        1: max(featAttrDict["workclass"], key = featAttrDict["workclass"].get),
        2: max(featAttrDict["fnlwgt"], key = featAttrDict["fnlwgt"].get),
        3: max(featAttrDict["education"], key = featAttrDict["education"].get),
        4: max(featAttrDict["education-num"], key = featAttrDict["education-num"].get),
        5: max(featAttrDict["marital-status"], key = featAttrDict["marital-status"].get),
        6: max(featAttrDict["occupation"], key = featAttrDict["occupation"].get),
        7: max(featAttrDict["relationship"], key = featAttrDict["relationship"].get),
        8: max(featAttrDict["race"], key = featAttrDict["race"].get),
        9: max(featAttrDict["sex"], key = featAttrDict["sex"].get),
        10: max(featAttrDict["capital-gain"], key = featAttrDict["capital-gain"].get),
        11: max(featAttrDict["capital-loss"], key = featAttrDict["capital-loss"].get),
        12: max(featAttrDict["hours-per-week"], key = featAttrDict["hours-per-week"].get),
        13: max(featAttrDict["native-country"], key = featAttrDict["native-country"].get)
    }

    for line in arr:
        i = 0
        for cell in line:
            if cell == "?":
                line[i] = majorityDict[i]
            i += 1


def FixTestData(testData):
    setFeatureAttrCount(testData, ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"])
    for key in featAttrDict:
        if "?" in featAttrDict[key]:
            del featAttrDict[key]["?"]

    majorityDict = {
        0: max(featAttrDict["age"], key = featAttrDict["age"].get),
        1: max(featAttrDict["workclass"], key = featAttrDict["workclass"].get),
        2: max(featAttrDict["fnlwgt"], key = featAttrDict["fnlwgt"].get),
        3: max(featAttrDict["education"], key = featAttrDict["education"].get),
        4: max(featAttrDict["education-num"], key = featAttrDict["education-num"].get),
        5: max(featAttrDict["marital-status"], key = featAttrDict["marital-status"].get),
        6: max(featAttrDict["occupation"], key = featAttrDict["occupation"].get),
        7: max(featAttrDict["relationship"], key = featAttrDict["relationship"].get),
        8: max(featAttrDict["race"], key = featAttrDict["race"].get),
        9: max(featAttrDict["sex"], key = featAttrDict["sex"].get),
        10: max(featAttrDict["capital-gain"], key = featAttrDict["capital-gain"].get),
        11: max(featAttrDict["capital-loss"], key = featAttrDict["capital-loss"].get),
        12: max(featAttrDict["hours-per-week"], key = featAttrDict["hours-per-week"].get),
        13: max(featAttrDict["native-country"], key = featAttrDict["native-country"].get)
    }

    for line in testData:
        i = 0
        for cell in line:
            if cell == "?":
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

# def overallMajority(dataset):
#     totalCount = len(dataset) * 1.0
#     labelCount = 0
#     minLabelCount = featAttrDict["y"]["yes"]
#     mError = 0

#     for label in featAttrDict["y"]:
#         if featAttrDict["y"][label] > 0:
#             labelCount += 1

#     if labelCount > 1:
#         for label in featAttrDict["y"]:
#             if featAttrDict["y"][label] < minLabelCount:
#                 minLabelCount = featAttrDict["y"][label]
#         mError = minLabelCount / totalCount
#     return mError


# def attributeMajority(dataset, attribute):
#     majorityError = 0
#     attrLabelDict = {
#         "yes": 0,
#         "no": 0
#     }
#     count = 0
#     for line in dataset:
#         if line[0] == attribute:
#             count += 1
#             attrLabelDict[line[1]] += 1

    
#     if count > 0:
#         minAttrVal = min(attrLabelDict.values()) * 1.0
#         majorityError = minAttrVal/count

#     return majorityError


# def featureMajority(data, featureName):
#     majorityError = 0
#     totalSize = len(data)*1.0
#     for attribute in featAttrDict[featureName]:
#         if featAttrDict[featureName][attribute] > 0:
#             majorityError += (featAttrDict[featureName][attribute]/totalSize) * attributeMajority(data, attribute)
#     return majorityError


# def findLowestMajorityFeature(dataset, colNameArr):
#     minMajorityError = 2
#     minMajorityErrorCol = colNameArr[0]
#     totalCol = len(dataset[0])
#     colCount = 0
#     for feature in colNameArr[:-1]:
#         featureLabelCol = dataset[:, [colCount, totalCol - 1]]
#         majorityError = featureMajority(featureLabelCol, feature)
#         if majorityError < minMajorityError:
#             minMajorityError = majorityError
#             minMajorityErrorCol = feature
#         colCount += 1
#     return minMajorityErrorCol

#Majority Error Calculation ------------------------------- End

#Gini Index Calculation ------------------------------- Start

# def overallGini(dataset):
#     totalCount = len(dataset)
#     labelCount = 0
#     giniIndex = 0

#     for label in featAttrDict["y"]:
#         if featAttrDict["y"][label] > 0:
#             labelCount += 1

#     if labelCount > 1:
#         total = 0
#         for label in featAttrDict["y"]:
#             probSquare = (featAttrDict["y"][label]/(totalCount * 1.0)) ** 2
#             total += probSquare
#         giniIndex = 1 - total

#     return giniIndex


# def attributeGini(dataset, attribute):
#     dataLen = len(dataset) * 1.0
#     gini = 0
#     attrLabelDict = {
#         "yes": 0,
#         "no": 0
#     }
#     count = 0
#     for line in dataset:
#         if line[0] == attribute:
#             count += 1
#             attrLabelDict[line[1]] += 1

#     if count > 0:
#         prob1 = (attrLabelDict["yes"]/count) ** 2
#         prob2 = (attrLabelDict["no"]/count) ** 2
#         probSum = prob1 + prob2
#         gini = 1 - probSum

#     return gini


# def featureGini(data, featureName):
#     gini = 0
#     totalSize = len(data)*1.0
#     for attribute in featAttrDict[featureName]:
#         if featAttrDict[featureName][attribute] > 0:
#             gini += (featAttrDict[featureName][attribute]/totalSize) * attributeGini(data, attribute)
#     return gini


# def findLowestGiniFeature(dataset, colNameArr):
#     minGini = 2
#     minGiniCol = colNameArr[0]
#     totalCol = len(dataset[0])
#     colCount = 0
#     for feature in colNameArr[:-1]:
#         featureLabelCol = dataset[:, [colCount, totalCol - 1]]
#         gini = featureGini(featureLabelCol, feature)
#         if gini < minGini:
#             minGini = gini
#             minGiniCol = feature
#         colCount += 1
#     return minGiniCol

#Gini Index Calculation ------------------------------- End

#ID3 Recursion ------------------------------------------- Start

def ID3DecisionEntropyTree(dataset, colNames, depth, maxDepth):
    setFeatureAttrCount(dataset, colNames)
    H = overallEntropy(dataset)
    if depth==maxDepth or H == 0:
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
            children[branch] = ID3DecisionEntropyTree(trimDataset, colNames, depth+1, maxDepth)

    return Node(colWithLowestEntropy, depth, False, children)


# def ID3DecisionMajorityTree(dataset, colNames, depth, maxDepth):
#     setFeatureAttrCount(dataset, colNames)
#     H = overallMajority(dataset)

#     if depth == maxDepth or H == 0:
#         return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
    
#     colWithLowestME = findLowestMajorityFeature(dataset, colNames)
#     colPos = np.where(colNames == colWithLowestME)[0][0]
#     colNames = np.delete(colNames, colPos)
#     children = {}
#     for branch in featAttrDict[colWithLowestME]:
#         trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
#         if len(trimDataset) == 0:
#             return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
#         trimDataset = np.delete(trimDataset, colPos, 1)
#         children[branch] = ID3DecisionMajorityTree(trimDataset, colNames, depth+1, maxDepth)

#     return Node(colWithLowestME, depth, False, children)


# def ID3DecisionGiniTree(dataset, colNames, depth, maxDepth):
#     setFeatureAttrCount(dataset, colNames)
#     H = overallGini(dataset)
#     if depth == maxDepth or H == 0:
#         return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
    
#     colWithLowestGini = findLowestGiniFeature(dataset, colNames)
#     colPos = np.where(colNames == colWithLowestGini)[0][0]
#     colNames = np.delete(colNames, colPos)
#     children = {}
#     for branch in featAttrDict[colWithLowestGini]:
#         trimDataset = list(filter(lambda x: x[colPos] == branch, dataset))
#         if len(trimDataset) == 0:
#             return Node("y", depth, True, max(featAttrDict["y"], key=featAttrDict["y"].get))
#         trimDataset = np.delete(trimDataset, colPos, 1)
#         children[branch] = ID3DecisionGiniTree(trimDataset, colNames, depth+1, maxDepth)

#     return Node(colWithLowestGini, depth, False, children)

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
    with open('test_final.csv', 'r') as f:
        next(f)
        for line in f:
            terms = line.strip().split(',')
            testdataList.append(terms)
    
    testArr = np.array(testdataList)
    testContArr = setContiniousValueTest(testArr[:,1:])

    #Comment this line to replace missing data with "unknown". Uncomment it replace missing data with majority value
    # testData = FixTestData(testContArr)

    with open("myfile.csv", "w") as w:
        w.write("ID,Prediction\n")
        id = 1
        for line in testContArr:
            prediction = TestPredict(line, rootNode, np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]))
            w.write(f'{id},{prediction}\n')
            id += 1
    
#Testing - Traversing the tree ---------------------------- Start            


#Driver Function -------------------------- Start

#["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "y"]
def DecisionDriver():
    setContiniousValue()
    setFeatureAttrCount(arr, ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "y"])

    #Comment this line to replace missing data with "unknown". Uncomment it replace missing data with majority value
    # FixTrainingData()
    # setFeatureAttrCount(arr, ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "y"])

    #Change the max depth value here to change the depth of the decision tree.

    for i in range(1, 17):
        maxDepth = i
        #Change the Function name to ID3DecisionEntropyTree or ID3DecisionMajorityTree or ID3DecisionGiniTree to use the respective algorithms.
        node = ID3DecisionEntropyTree(arr, np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "y"]), 0, maxDepth)
        TestData(node)

DecisionDriver()

#Driver Function -------------------------- End