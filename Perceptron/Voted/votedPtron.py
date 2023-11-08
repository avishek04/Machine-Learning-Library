import numpy as np
import pandas as pd

def TestData(dataFrame, wtVectorList):
    X_test, Y_test = GetXYFrame(dataFrame)
    errorCount = 0
    dataLen = len(dataFrame)
    weights = len(wtVectorList)

    for i in range(dataLen):
        yVal = 2 * Y_test[i] - 1
        dotSum = 0
        for j in range(weights):
            dotVal = np.dot(wtVectorList[j]["wt"], X_test[i])
            dotSum += (-wtVectorList[j]["c"]) if dotVal < 0 else (wtVectorList[j]["c"])
        errorCount += 1 if yVal * dotSum < 0 else 0

    print (f'Average Prediction Error: {errorCount/dataLen}')

def GetXYFrame(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    return X, Y


def VotedPerceptron(dataFrame, epoch):
    wtVectorLen = len(dataFrame.axes[1]) - 1
    wtVector = np.zeros(wtVectorLen)
    dataLen = len(dataFrame)
    wtVectorList = []
    r = 1
    c = 0

    for i in range(epoch):
        data = dataFrame.sample(frac = 1)
        X_train, Y_train = GetXYFrame(data)
        
        for j in range(dataLen):
            yVal = 2 * Y_train[j] - 1
            if (yVal * np.dot(wtVector, X_train[j]) <= 0):
                print(f'Weight Vector:{wtVector}, Count: {c}')
                wtDict = {
                    "wt": wtVector,
                    "c": c
                }
                wtVectorList.append(wtDict)
                wtVector += r * yVal * X_train[j]
                c = 1
            else:
                c += 1
    return wtVectorList

    
trainPath = 'train.csv'
dfTrain = pd.read_csv(trainPath)
epoch = 10

weightVectorList = VotedPerceptron(dfTrain, epoch)

testPath = "test.csv"
dfTest = pd.read_csv(testPath)
TestData(dfTest, weightVectorList)