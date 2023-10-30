import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cost_function(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error**2)
    return cost

def batch_gradient_descent(X, y, lr=1, tolerance=1e-6):
    m, n = X.shape
    weights = np.zeros(n)
    costs = []

    while True:
        predictions = X.dot(weights)
        error = predictions - y
        gradient = (1/m) * X.T.dot(error)
        weights -= lr * gradient

        current_cost = cost_function(X, y, weights)
        costs.append(current_cost)

        # Check convergence
        if len(costs) > 1 and np.linalg.norm(weights - previous_weights) < tolerance:
            break

        previous_weights = np.copy(weights)
        lr /= 2

    return weights, costs, lr

X_train = ...  
y_train = ...  
X_test = ...   
y_test = ...   

trainPath = 'train.csv'
dfTrain = pd.read_csv(trainPath)
X_train = dfTrain.iloc[:, :-1].values
y_train = dfTrain.iloc[:, -1].values

testPath = 'test.csv'
dfTest = pd.read_csv(testPath)
X_test = dfTest.iloc[:, :-1].values
y_test = dfTest.iloc[:, -1].values

# Initialize learning rate
learning_rate = 1
weights, costs, learning_rate = batch_gradient_descent(X_train, y_train, learning_rate)

for cost in costs:
    print(cost)

test_cost = cost_function(X_test, y_test, weights)
print(f"Final Weight Vector: {weights}")
print(f"Learning Rate: {learning_rate}")
print(f"Cost on Test Data: {test_cost}")

# Plot cost function changes
plt.plot(costs, label=f'Learning Rate: {learning_rate}')
plt.xlabel('Iterations')
plt.ylabel('Cost Function Value')
plt.legend()
plt.show()

# Use the final weights to calculate the cost function value on test data

