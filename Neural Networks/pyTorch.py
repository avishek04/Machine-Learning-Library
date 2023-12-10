import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class NeuralNetTorch(nn.Module):
    def __init__(self, inputSize, hiddenLayers, hiddenSize, outputSize, activeFunc="tanh"):
        super().__init__()
        if activeFunc == "relu":
            self.hidLayer = nn.ModuleList([])
            self.hidLayer.append(nn.Sequential(nn.Linear(inputSize, hiddenSize), nn.ReLU()))

            for i in range(hiddenLayers - 1):
                self.hidLayer.append(nn.Sequential(nn.Linear(hiddenSize, hiddenSize), nn.ReLU()))
        else:
            self.hidLayer = nn.ModuleList([])
            self.hidLayer.append(nn.Sequential(nn.Linear(inputSize, hiddenSize), nn.Tanh()))

            for i in range(hiddenLayers - 1):
                self.hidLayer.append(nn.Sequential(nn.Linear(hiddenSize, hiddenSize), nn.Tanh()))

        self.outLayer = nn.Linear(hiddenSize, outputSize)


    @staticmethod
    def xavierIn(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


    @staticmethod
    def hInitialize(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
            

    def trainModel(self, lossFunc, opt, epochs=20):
        lossH = [0] * epochs
        acc = [0] * epochs

        for i in range(epochs):
            self.train()

            for x, y in train_dataset:
                prediction = self(x)
                prediction = prediction.reshape(prediction.shape[0], )
                loss = lossFunc(prediction, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
                lossH[i] += loss.item() * y.size(0)
                isCorrect = (torch.argmax(prediction) == y).float()
                acc[i] += isCorrect.mean()

            lossH[i] /= len(train_dataset.dataset)
            acc[i] /= len(train_dataset.dataset)


    def forward(self, x_train):
        for ly in self.hidLayer:
            x_train = ly(x_train)

        return self.outLayer(x_train)


    def calculateError(self, actual, predicted):
        predicted[predicted > 0.5] = 1
        predicted[predicted < 0.5] = 0
        predicted = predicted.reshape(len(predicted), )
        correct = torch.sum(predicted == actual)
        return 1 - correct.item() / len(actual)


    def predict(self, X_test):
        return self(X_test)


if __name__ == "__main__":
    X_train = pd.read_csv('train.csv', header=None)
    X_test = pd.read_csv('test.csv', header=None)

    y = np.array(X_train.iloc[:, 4])
    X_train = np.array(X_train.iloc[:, :4])

    y_test = np.array(X_test.iloc[:, 4])
    X_test = np.array(X_test.iloc[:, :4])

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    dataset = TensorDataset(X_train, y_train)

    train_dataset = DataLoader(dataset, batch_size=10, shuffle=True)

    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]

    print("RELU:\n")

    for depth in depths:
        for width in widths:
            print(f"Depth: {depth} Width: {width}")
            inputSize = X_train.shape[1]
            hiddenSize = width
            outputSize = 1
            hiddenLayers = depth
            nnmodel = NeuralNetTorch(inputSize, hiddenLayers, hiddenSize, outputSize, "relu")
            nnmodel.apply(NeuralNetTorch.hInitialize)
            trainModel = nn.MSELoss()
            optimiser = torch.optim.Adam(nnmodel.parameters(), lr=0.001)
            nnmodel.trainModel(trainModel, optimiser)
            predicted_train = nnmodel.predict(X_train)
            predicted_test = nnmodel.predict(X_test)

            print("Training error: ", nnmodel.calculateError(y_train, predicted_train))
            print("Test error: ", nnmodel.calculateError(y_test, predicted_test))

    print("\nTANH:\n")

    for depth in depths:
        for width in widths:
            print(f"Depth: {depth} Width: {width}")
            input_size = X_train.shape[1]
            hidden_size = width
            output_size = 1
            no_hidden_layers = depth
            nnmodel = NeuralNetTorch(inputSize, hiddenLayers, hiddenSize, outputSize)
            nnmodel.apply(NeuralNetTorch.xavierIn)
            trainModel = nn.MSELoss()
            optimiser = torch.optim.Adam(nnmodel.parameters(), lr=0.001)
            nnmodel.trainModel(trainModel, optimiser)
            predicted_train = nnmodel.predict(X_train)
            predicted_test = nnmodel.predict(X_test)

            print("Training error: ", nnmodel.calculateError(y_train, predicted_train))
            print("Test error: ", nnmodel.calculateError(y_test, predicted_test))