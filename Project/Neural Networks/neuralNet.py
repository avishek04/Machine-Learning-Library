import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layers=2, hunits=(2, 2)):
        self.loss = []
        self.neuralnet = []
        self.layers = layers
        self.hlunits = hunits
        self.finalOutput = None
          

    def createNeuralNet(self, inputs):
        for i in range(self.layers):
            hl = [{'weights': np.array([random.gauss(mu=0.0, sigma=1.0) for i in range(inputs + 1)])} for j in range(self.hlunits[i])]
            inputs = self.hlunits[i]
            self.neuralnet.append(hl)

        outLayer = [{'weights': np.array([random.gauss(mu=0.0, sigma=1.0) for i in range(self.hlunits[self.layers - 1] + 1)])}]
        self.neuralnet.append(outLayer)


    def sigmoid(self, value):
        return value * (1 - value)
    

    def neuronOut(self, weights, input, layer):
        sum = weights[-1]

        for i in range(len(weights) - 1):
            sum += weights[i] * input[i]

        if layer == len(self.neuralnet) - 1:
            return sum
        
        return 1 / (1 + np.exp(-sum))


    def backProp(self, actualValue, lr):
        for i in reversed(range(len(self.neuralnet))):
            layer = self.neuralnet[i]

            if i == len(self.neuralnet) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    error = neuron['output'] - actualValue
                    neuron['cache'] = error
                    neuron['input'] = np.append(neuron['input'], [1])
                    neuron['weights'] -= lr * neuron['cache'] * neuron['input']
            else:
                for j in range(len(layer)):
                    cache = np.zeros(shape=(1,), dtype=float)

                    for neuron in self.neuralnet[i + 1]:
                        cache[0] += neuron['weights'][j] * neuron['cache']

                    layer[j]['cache'] = cache[0]
                    layer[j]['input'] = np.append(layer[j]['input'], [1])
                    layer[j]['weights'] -= lr * layer[j]['cache'] * self.sigmoid(layer[j]['output']) * \
                                           layer[j]['input']
                    

    def forward(self, inputVal):
        for layer in range(len(self.neuralnet)):
            out = []

            for neuron in self.neuralnet[layer]:
                neuron['output'] = self.neuronOut(neuron["weights"], inputVal, layer)
                neuron['input'] = np.array(inputVal)
                out.append(neuron['output'])

            inputVal = out

        self.finalOutput = inputVal
        return inputVal
    
    
    def sgdNN(self, X, y, lr=None, epoch=20):
        for i in range(epoch):
            index = random.sample(range(len(X)), len(X))

            for j in index:
                row = X[j]
                self.forward(row)
                expected = y.iloc[j]
                self.backProp(expected, lr(i))


    def zeroWeight(self, noInputs):
        for i in range(self.layers):
            hl = [{'weights': np.array([0.0 for i in range(noInputs + 1)])} for j in range(self.hlunits[i])]
            noInputs = self.hlunits[i]
            self.neuralnet.append(hl)

        outLayer = [{'weights': np.array([0.0 for i in range(self.hlunits[self.layers - 1] + 1)])}]
        self.neuralnet.append(outLayer)


    def getError(self, actual, prediction):
        return 1 - (np.sum(actual == prediction) / len(actual))
    

    def predict(self, X_test):
        X = np.array(X_test)
        output = lambda data: self.forward(data)
        prediction = np.array([output(data)[0] for data in X])
        newOut = prediction.copy()
        newOut[prediction < 0.5] = 0
        newOut[prediction > 0.5] = 1

        return newOut
    

if __name__ == "__main__":
    X_train = pd.read_csv('train.csv', header=None)
    X_test = pd.read_csv('test.csv', header=None)

    yTrain = X_train.iloc[:, 4]
    X_train = X_train.iloc[:, :4]
    y_test = X_test.iloc[:, 4]
    X_test = X_test.iloc[:, :4]
    
    T = [x for x in range(15)]
    widths = [5, 10, 25, 50, 100]
    gamma = 0.01

    for width in widths:
        lRate = lambda i: gamma / (1 + (gamma * i) / width)
        nn = NeuralNetwork(layers=2, hunits=(width, width))

        nn.createNeuralNet(X_train.shape[1])
        nn.sgdNN(np.array(X_train), yTrain, lRate)

        trainPredict = nn.predict(X_train)
        testPredict = nn.predict(X_test)

        print("Width: ", width)
        print("Training Error: ", nn.getError(yTrain, trainPredict))
        print("Test Error: ", nn.getError(y_test, testPredict))

    print("\nAll weights set to zero\n")

    for width in widths:
        learning_rate = lambda i: gamma / (1 + (gamma * i) / width)
        nn = NeuralNetwork(layers=2, hunits=(width, width))

        nn.zeroWeight(X_train.shape[1])
        nn.sgdNN(np.array(X_train), yTrain, lRate)

        trainPredict = nn.predict(X_train)
        testPredict = nn.predict(X_test)

        print("Width: ", width)
        print("Training Error: ", nn.getError(yTrain, trainPredict))
        print("Test Error: ", nn.getError(y_test, testPredict))