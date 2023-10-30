import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def fit(self, X, y, weights):
        num_samples, num_features = X.shape
        assert len(y) == num_samples
        assert len(weights) == num_samples

        best_error = float('inf')

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                predictions = np.ones_like(y)
                predictions[X[:, feature_index] < threshold] = -1

                error = np.sum(weights * (predictions != y))

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold

        self.alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
        return best_error

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.ones(num_samples)
        predictions[X[:, self.feature_index] < self.threshold] = -1
        return predictions


def adaboost(X_train, y_train, X_test, y_test, num_iterations):
    num_samples, _ = X_train.shape
    weights = np.ones(num_samples) / num_samples
    models = []
    train_errors = []
    test_errors = []
    train_stump_errors = []

    for _ in range(num_iterations):
        # Create and fit a decision stump
        model = DecisionStump()
        train_stump_error = model.fit(X_train, y_train, weights)
        train_stump_errors.append(train_stump_error)

        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Compute errors
        train_error = np.sum(weights * (train_predictions != y_train))
        test_error = np.sum(test_predictions != y_test)

        # Update weights
        incorrect = (train_predictions != y_train)
        error = np.sum(weights[incorrect])
        alpha = model.alpha
        weights *= np.exp(alpha * incorrect)

        # Normalize weights
        weights /= np.sum(weights)

        # Save model and errors
        models.append(model)
        train_errors.append(train_error / num_samples)
        test_errors.append(test_error / len(y_test))

    return models, train_errors, test_errors

def setContiniousValue(inptArr):
    for line in inptArr:
        if (int(line[0]) < 100000):
            line[0] = "low"
        elif (int(line[0]) < 300000):
            line[0] = "med" 
        else:
            line[0] = "high"

        if (int(line[2]) > 3 or int(line[2]) < 1):
            line[2] = "4"

        if (int(line[3]) > 2):
            line[3] = "3"

        if (int(line[4]) < 30):
            line[4] = "low"
        elif (int(line[11]) < 50):
            line[4] = "med" 
        else:
            line[4] = "high"

        if (int(line[11]) < 10000):
            line[11] = "low"
        elif (int(line[11]) < 100000):
            line[11] = "med" 
        else:
            line[11] = "high"
        
        if (int(line[12]) < 10000):
            line[12] = "low"
        elif (int(line[12]) < 100000):
            line[12] = "med" 
        else:
            line[12] = "high"

        if (int(line[13]) < 10000):
            line[13] = "low"
        elif (int(line[13]) < 100000):
            line[13] = "med" 
        else:
            line[13] = "high"

        if (int(line[14]) < 10000):
            line[14] = "low"
        elif (int(line[14]) < 100000):
            line[14] = "med" 
        else:
            line[14] = "high"

        if (int(line[15]) < 10000):
            line[15] = "low"
        elif (int(line[15]) < 100000):
            line[15] = "med" 
        else:
            line[15] = "high"

        if (int(line[16]) < 10000):
            line[16] = "low"
        elif (int(line[16]) < 100000):
            line[16] = "med" 
        else:
            line[16] = "high"

        if (int(line[17]) < 3000):
            line[17] = "low"
        elif (int(line[17]) < 10000):
            line[17] = "med"
        else:
            line[17] = "high"

        if (int(line[18]) < 3000):
            line[18] = "low"
        elif (int(line[18]) < 10000):
            line[18] = "med"
        else:
            line[18] = "high"

        if (int(line[19]) < 3000):
            line[19] = "low"
        elif (int(line[19]) < 10000):
            line[19] = "med"
        else:
            line[19] = "high"

        if (int(line[20]) < 3000):
            line[20] = "low"
        elif (int(line[20]) < 10000):
            line[20] = "med"
        else:
            line[20] = "high"

        if (int(line[21]) < 3000):
            line[21] = "low"
        elif (int(line[21]) < 10000):
            line[21] = "med"
        else:
            line[21] = "high"

        if (int(line[22]) < 3000):
            line[22] = "low"
        elif (int(line[22]) < 10000):
            line[22] = "med"
        else:
            line[22] = "high"

    return inptArr


def setYValue(input):
    count = 0
    for value in input:
        if value == "1":
            input[count] = 1
        else:
            input[count] = -1
        count += 1
    return input

X_train = ...  
y_train = ...  
X_test = ...   
y_test = ...   

data = []
with open('creditCardTrain.csv', 'r') as f:
        next(f)
        for line in f:
            terms = line.strip().split(',')
            data.append(terms)

random.shuffle(data)
data = setContiniousValue(data)
arr = pd.DataFrame(data[:24000])
testArr = pd.DataFrame(data[24000:])

X_train = arr.iloc[:, :-1].values
y_train = setYValue(arr.iloc[:, -1].values)

X_test = testArr.iloc[:, :-1].values
y_test = setYValue(testArr.iloc[:, -1].values)

num_iterations = 500
models, train_errors, test_errors = adaboost(X_train, y_train, X_test, y_test, num_iterations)

# Plotting the errors
plt.figure(figsize=(12, 5))
plt.plot(range(1, num_iterations + 1), train_errors, label='Training Error')
plt.plot(range(1, num_iterations + 1), test_errors, label='Test Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Iterations')
plt.legend()
plt.tight_layout()
plt.show()
