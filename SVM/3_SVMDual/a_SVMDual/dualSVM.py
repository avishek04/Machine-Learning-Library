import numpy as np
import os
from scipy.optimize._minimize import minimize

def objective(a, inputs,outputs):
    alphaSum = np.sum(a)
    xMat = np.asmatrix(inputs)
    x = xMat * np.transpose(xMat)
    yMat = np.outer(a,outputs)
    y = yMat*np.transpose(yMat)
    sum = np.sum(y*x)
    return (1/2)*(sum - alphaSum)

def Labels(data):
    rowLen = len(data[0]) - 1
    for line in data:
        if line[rowLen] == 0:
            line[rowLen] = -1
    return data

def DualSVM(inputs,outputs,c,a):
    bound = [(0,c)]*len(a)
    cons = [{'type':'eq','fun': lambda a: np.dot(a,outputs),'jac': lambda a: outputs}]
    sol = minimize(fun=objective, x0=a, args=(inputs,outputs), method='SLSQP', constraints=cons, bounds=bound)
    return sol

def PredictionError(weight_vector,test_data):
    error = 0
    for example in test_data:
        inputs = example[slice(0,len(example)-1)]
        prediction = np.sign(np.dot(weight_vector[0:4],inputs)+weight_vector[4])
        if prediction != example[len(example)-1]:
            error += 1
    return error/len(test_data)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(here,'train.csv')
    test_file = os.path.join(here,'test.csv')
    data = np.genfromtxt(train_file,delimiter=',',dtype=np.float32)
    testData = np.genfromtxt(test_file,delimiter=',',dtype=np.float32)
    data = Labels(data)
    testData = Labels(testData)

    c_arr = [100/873,500/873,700/873]
    alphas = np.random.rand(len(data))
    biasData = np.insert(data,4,1,axis=1)
    inputs = []
    outputs = []

    for example in biasData:
        inputs.append(example[0:5])
        outputs.append(example[len(example)-1])

    for i in range(0,3):
        c = c_arr[i]
        mini = DualSVM(inputs,outputs,c,alphas)
        filterArr = []

        for alpha in mini.x:
            if alpha > 0.001:
                filterArr.append(True)
            else:
                filterArr.append(False)
        result = mini.x[filterArr]

        wtVector = sum(result[i]*inputs[i]*outputs[i] for i in range(len(result)))
        print(f'Weight Vector for c{i + 1}: {wtVector}')
        trainError = PredictionError(wtVector,data)
        print(f'Training Error for c{i + 1}: {trainError}')
        testError = PredictionError(wtVector,testData)
        print(f'Test Error for c{i + 1}: {testError}')

if __name__ == "__main__":
    main()