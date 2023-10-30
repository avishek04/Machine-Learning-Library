import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    test_stump_errors = []

    for _ in range(num_iterations):
        # Create and fit a decision stump
        model = DecisionStump()
        train_stump_error = model.fit(X_train, y_train, weights)
        train_stump_errors.append(train_stump_error)

        # test_stump_error = model.fit(X_test, y_test, weights)
        # test_stump_errors.append(test_stump_error)

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

    return models, train_errors, test_errors, train_stump_errors, test_stump_errors

def setContiniousValue(inptArr):
    for line in inptArr:
        #age
        if (int(line[0]) > 25) and (int(line[0]) < 50):
            line[0] = "yes"
        else:
            line[0] = "no" 
        #balance
        if int(line[5]) > 500:
            line[5] = "yes"
        else:
            line[5] = "no"
        #day
        if int(line[9]) > 15:
            line[9] = "yes"
        else:
            line[9] = "no"
        #duration
        if int(line[11]) > 180:
            line[11] = "no"
        else:
            line[11] = "yes"
        #campaign
        if int(line[12]) > 2:
            line[12] = "no"
        else:
            line[12] = "yes"
        #pdays
        if (int(line[13]) == -1) or (int(line[13]) < 200):
            line[13] = "no"
        else:
            line[13] = "yes"
        #previous
        if int(line[14]) > 2:
            line[14] = "yes"
        else:
            line[14] = "no"
    return inptArr

def setYValue(input):
    count = 0
    for value in input:
        if value == "yes":
            input[count] = 1
        else:
            input[count] = -1
        count += 1
    return input

X_train = ...  
y_train = ...  
X_test = ...   
y_test = ...   

trainPath = 'train.csv'
dfTrain = pd.read_csv(trainPath)
X_train = setContiniousValue(dfTrain.iloc[:, :-1].values)
y_train = setYValue(dfTrain.iloc[:, -1].values)

testPath = 'test.csv'
dfTest = pd.read_csv(testPath)
X_test = setContiniousValue(dfTest.iloc[:, :-1].values)
y_test = setYValue(dfTest.iloc[:, -1].values)

num_iterations = 500
models, train_errors, test_errors, train_stump_errors, test_stump_errors = adaboost(X_train, y_train, X_test, y_test, num_iterations)

# Plotting the errors
plt.figure(figsize=(12, 5))

# Plotting the errors for each iteration
plt.plot(range(1, num_iterations + 1), train_errors, label='Training Error')
plt.plot(range(1, num_iterations + 1), test_errors, label='Test Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Iterations')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the errors for each decision stump
plt.plot(range(1, num_iterations + 1), train_stump_errors, label='Train Stump Error', color='red')
# plt.plot(range(1, num_iterations + 1), test_stump_errors, label='Test Stump Error', color='blue')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.title('Decision Stump Errors vs. Iterations')
plt.legend()
plt.tight_layout()
plt.show()

