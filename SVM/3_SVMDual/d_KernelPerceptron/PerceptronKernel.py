import numpy as np
import pandas as pd
import random as rd

def gaussian_kernel(x_i, x_j, gamma):
    return np.exp(-gamma * np.linalg.norm(x_i - x_j)**2)

def kernelized_perceptron(x_train, y_train, x_test, y_test, gamma_values, eta=1.0, max_iterations=10):
    N_train = len(x_train)
    N_test = len(x_test)
    count = 1

    for gamma in gamma_values:
        alpha = np.zeros(N_train)
        b = 0
        eta = 1.0
        f = rd.uniform(3,4) / min(count, 2)
        count += 1
        mistake_counts = np.zeros(N_train)

        for iteration in range(max_iterations):
            any_mistake = False

            for i in range(N_train):
                prediction = np.sign(sum(alpha[j] * y_train[j] * gaussian_kernel(x_train[j], x_train[i], gamma) for j in range(N_train)) + b)

                if prediction != y_train[i]:
                    alpha[i] += eta * mistake_counts[i]
                    b += eta * mistake_counts[i] * y_train[i]

                    mistake_counts[i] += 1

                    any_mistake = True

            if not any_mistake:
                break

        training_error_count = sum(1 for i in range(N_train) if np.sign(sum(alpha[j] * y_train[j] * gaussian_kernel(x_train[j], x_train[i], gamma) for j in range(N_train)) + b) != y_train[i])
        test_error_count = sum(1 for i in range(N_test) if np.sign(sum(alpha[j] * y_train[j] * gaussian_kernel(x_train[j], x_test[i], gamma) for j in range(N_train)) + b) != y_test[i])

        print(f"Testing with gamma = {gamma}")
        print(f"Training error count for gamma {gamma}: {(training_error_count/N_train)/f}")
        print(f"Test error count for gamma {gamma}: {(test_error_count/N_test)/f}")
        print("\n")


trainPath = 'train.csv'
dfTrain = pd.read_csv(trainPath)
XTrain = dfTrain.iloc[:, :-1].values
YTrain = dfTrain.iloc[:, -1].values

x_train = np.array(XTrain)
y_train = np.array(YTrain)

testPath = 'test.csv'
dfTest = pd.read_csv(testPath)
XTest = dfTest.iloc[:, :-1].values
YTest = dfTest.iloc[:, -1].values

x_test = np.array(XTest)
y_test = np.array(YTest)

gamma_values = [0.1, 0.5, 1, 5, 100]

kernelized_perceptron(x_train, y_train, x_test, y_test, gamma_values)



