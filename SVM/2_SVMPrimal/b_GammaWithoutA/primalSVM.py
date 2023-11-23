import numpy as np
import pandas as pd

def TestData(dataFrame, weightVector):
    X_test, Y_test = GetXYFrame(dataFrame)
    X_test = np.insert(X_test, len(weightVector) - 1, 1, axis=1)
    errorCount = 0
    dataLen = len(dataFrame)

    for i in range(dataLen):
        yVal = 2 * Y_test[i] - 1
        errorCount += 1 if yVal * np.dot(weightVector, X_test[i]) < 0 else 0

    return errorCount/dataLen


def GetXYFrame(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    return X, Y


def PrimalSVM(dataFrame, epoch, C):
    wtVectorLen = len(dataFrame.axes[1]) - 1
    wtVector = np.zeros(wtVectorLen + 1)
    dataLen = len(dataFrame)
    gamma0 = 1

    for i in range(epoch):
        data = dataFrame.sample(frac = 1)
        X_train, Y_train = GetXYFrame(data)
        X_train = np.insert(X_train, wtVectorLen, 1, axis=1)
        gammaT = gamma0 / (1 + i)

        for j in range(dataLen):
            yVal = 2 * Y_train[j] - 1
            if (yVal * np.dot(wtVector, X_train[j]) <= 1):
                tempWtVector = wtVector
                tempWtVector[wtVectorLen] = 0
                wtVector = wtVector - gammaT * tempWtVector + gammaT * C * dataLen * yVal * X_train[j]
            else:
                b = wtVector[wtVectorLen]
                wtVector = (1 - gammaT) * wtVector
                wtVector[wtVectorLen] = b

    # print (f'Learned Weight Vector: {wtVector}')
    return wtVector

    
trainPath = 'train.csv'
dfTrain = pd.read_csv(trainPath)
epoch = 100
c1 = 100/873
c2 = 500/873
c3 = 700/873

testPath = "test.csv"
dfTest = pd.read_csv(testPath)

weightVectorC1 = PrimalSVM(dfTrain, epoch, c1)
trainErrorC1 = TestData(dfTrain, weightVectorC1)
testErrorC1 = TestData(dfTest, weightVectorC1)

print(f"Weight Vector for C1: {weightVectorC1}")
print(f"Training Error for C1: {trainErrorC1}")
print(f"Test Error for C1: {testErrorC1}\n")

weightVectorC2 = PrimalSVM(dfTrain, epoch, c2)
trainErrorC2 = TestData(dfTrain, weightVectorC2)
testErrorC2 = TestData(dfTest, weightVectorC2)

print(f"Weight Vector for C2: {weightVectorC2}")
print(f"Training Error for C2: {trainErrorC2}")
print(f"Test Error for C2: {testErrorC2}\n")

weightVectorC3 = PrimalSVM(dfTrain, epoch, c3)
trainErrorC3 = TestData(dfTrain, weightVectorC3)
trainErrorC3 = TestData(dfTest, weightVectorC3)

print(f"Weight Vector for C3: {weightVectorC3}")
print(f"Training Error for C3: {trainErrorC3}")
print(f"Test Error for C3: {trainErrorC3}")