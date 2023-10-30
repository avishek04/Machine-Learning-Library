import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cost_function(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error**2)
    return cost

def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    costs = []

    for epoch in range(num_epochs):
        for i in range(m):
            # Randomly select a training example
            random_index = np.random.randint(0, m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # Calculate stochastic gradient
            prediction = xi.dot(weights)
            error = prediction - yi
            gradient = xi.T.dot(error)

            # Update weights
            weights -= learning_rate * gradient.flatten()

            # Calculate and store cost after each update
            current_cost = cost_function(X, y, weights)
            costs.append(current_cost)

    return weights, costs


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
learning_rate = 0.01

# Perform stochastic gradient descent
final_weights, costs = stochastic_gradient_descent(X_train, y_train, learning_rate)

test_cost = cost_function(X_test, y_test, final_weights)

for cost in costs:
    print(cost)
print(f"Learned Weight Vector: {final_weights}")
print(f"Chosen Learning Rate: {learning_rate}")
print(f"Cost on Test Data: {test_cost}")

# Plot cost function changes
plt.plot(costs)
plt.xlabel('Number of Updates')
plt.ylabel('Cost Function Value')
plt.title('Cost Function Values during SGD')
plt.show()
