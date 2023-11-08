import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def TestData(dataFrame, wtVector):
    X_test, Y_test = GetXYFrame(dataFrame)
    errorCount = 0
    dataLen = len(dataFrame)

    for i in range(dataLen):
        yVal = 2 * Y_test[i] - 1
        dotVal = np.dot(wtVector, X_test[i])
        errorCount += 1 if yVal * dotVal < 0 else 0

    print (f'Average Prediction Error: {errorCount/dataLen}')

def GetXYFrame(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    return X, Y


def AveragePerceptron(dataFrame, epoch):
    wtVectorLen = len(dataFrame.axes[1]) - 1
    wtVector = np.zeros(wtVectorLen)
    finalVector = np.zeros(wtVectorLen)
    dataLen = len(dataFrame)
    r = 1
    c = 0

    for i in range(epoch):
        data = dataFrame.sample(frac = 1)
        X_train, Y_train = GetXYFrame(data)
        
        for j in range(dataLen):
            yVal = 2 * Y_train[j] - 1
            if (yVal * np.dot(wtVector, X_train[j]) <= 0):
                wtVector += r * yVal * X_train[j]
            finalVector += wtVector 
    
    print(f'Learned Weight Vector: {finalVector}')
    return finalVector

    
trainPath = 'train.csv'
dfTrain = pd.read_csv(trainPath)
epoch = 10

weightVector = AveragePerceptron(dfTrain, epoch)

testPath = "test.csv"
dfTest = pd.read_csv(testPath)
TestData(dfTest, weightVector)