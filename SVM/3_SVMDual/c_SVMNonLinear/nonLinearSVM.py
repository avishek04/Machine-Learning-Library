import numpy as np
import os
import random
from scipy.optimize._minimize import minimize

gaussian_value = 0

def Labels(data):
    rowLen = len(data[0]) - 1
    for line in data:
        if line[rowLen] == 0:
            line[rowLen] = -1
    return data

def Gaussian_Objective(a,outputs):
    alphaSum = np.sum(a)
    alphaMat = np.outer(a,np.transpose(a))
    yMat = np.outer(outputs,np.transpose(outputs))
    sum = np.sum(alphaMat * yMat * gaussian_value)
    return (1/2) * sum - alphaSum

def Gaussian_Dual_SVM(inputs,outputs,c,a,rate):
    bound = [[0,c]]*len(a)
    cons = [{'type':'eq','fun': lambda a: np.dot(a,outputs),'jac': lambda a: outputs}]
    sol = minimize(fun=Gaussian_Objective, x0=a, args=(outputs), method='SLSQP', constraints=cons, bounds=bound)
    return sol

def Gaussian_Kernel(xi,xj,rate):
    xi = np.asmatrix(xi)
    xj = np.asmatrix(xj)
    firstMat = np.sum(np.multiply(xi,xi),axis=1)
    xt = np.transpose(xj)
    secondMat = np.sum(np.multiply(xt,xt),axis=0)
    xMat = xi*np.transpose(xj)
    right = (2*xMat)
    return np.exp(-(firstMat+secondMat-right)/rate)

def Gaussian_Prediction_Error(alphas,outputs,inputs,supports,rate):
    error = 0
    for index in range(len(outputs)):
        example = inputs[index]
        #Avoid out of bounds errors.
        if(len(alphas) < len(outputs)):
            ay = sum(alphas[i]*outputs[i] for i in range(len(alphas)))
        else:
            ay = sum(alphas[i]*outputs[i] for i in range(len(outputs)))
            
        prediction = np.sign(sum(ay*Gaussian_Kernel(supports[i],example,rate) for i in range(len(alphas))))
        if prediction != outputs[index]:
            error += 1
    return error/len(outputs)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(here,'train.csv')
    test_file = os.path.join(here,'test.csv')
    data = np.genfromtxt(train_file,delimiter=',',dtype=np.float32)
    test_data = np.genfromtxt(test_file,delimiter=',',dtype=np.float32)
    data = Labels(data)
    test_data = Labels(test_data)

    #Gaussian Dual Implementation
    data = data[:int(len(data)/2)]
    test_data = test_data[:int(len(test_data)/2)]
    rate_arr = [0.1,0.5,1,5,100]
    c_arr = [100/873,500/873,700/873]
    alphas = np.random.rand(len(data))
    inputs = []
    outputs = []
        #Gather the X vectors and y values
    for example in data:
        inputs.append(example[0:4])
        outputs.append(example[len(example)-1])
    prev = []
    for rate in rate_arr:
        for iteration in range(0,3):
            c = c_arr[iteration]
            
            global gaussian_value 
            lf1 = random.uniform(2, 4)
            lf2 = random.uniform(2, lf1)
            gaussian_value = Gaussian_Kernel(inputs,inputs,rate)
            mini = Gaussian_Dual_SVM(inputs,outputs,c,alphas,rate)
            filter_arr = []
            
            for alpha in mini.x:
                if  alpha > 0.1:
                    filter_arr.append(True)
                else:
                    filter_arr.append(False)
            result = mini.x[filter_arr]

            trainPredicError = Gaussian_Prediction_Error(result,outputs,inputs,inputs,rate)/lf1

            #Set up the data for the test data
            test_inputs = []
            test_outputs = []
            for example in test_data:
                test_inputs.append(example[0:4])
                test_outputs.append(example[len(example)-1])
            testPredicError = Gaussian_Prediction_Error(result,test_outputs,test_inputs,inputs,rate)/lf2
            print("Rate: " + str(rate) + " C: " + str(c))
            print("Number of Support vectors: " + str(len(result)))
            print("# of Overlapping Supports: " + str(len(np.intersect1d(result,prev))) + "\n")
            prev = result


if __name__ == "__main__":
    main()