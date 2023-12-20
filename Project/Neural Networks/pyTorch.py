import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

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

    @staticmethod
    def setContiniousValue(arr):
        for line in arr:
            #age
            line0 = int(line[0])
            if line0 < 25:
                line[0] = "veryLow"
            elif line0 < 40:
                line[0] = "low"
            elif line0 < 60:
                line[0] = "med" 
            else:
                line[0] = "high" 

            #fnlwgt
            line2 = int(line[2])
            if line2 < 100000:
                line[2] = "low"
            elif line2 < 350000:
                line[2] = "med"
            else:
                line[2] = "high"

            #education-num
            line4 = int(line[4])
            if line4 < 8:
                line[4] = "low"
            elif line4 < 12:
                line[4] = "med"
            else:
                line[4] = "high"

            #capital-gain
            line10 = int(line[10])
            if line10 < 1000:
                line[10] = "low"
            elif line10 < 10000:
                line[10] = "med"
            else:
                line[10] = "high"

            #capital-loss
            line11 = int(line[11])
            if line11 < 1000:
                line[11] = "low"
            else:
                line[11] = "high"

            #hours-per-week
            line12 = int(line[12])
            if line12 < 25:
                line[12] = "low"
            elif line12 < 45:
                line[12] = "med"
            else:
                line[12] = "high"

        return arr

if __name__ == "__main__":
    X_train = pd.read_csv('Neural Networks/train.csv', skiprows=1, header=None)
    X_test = pd.read_csv('Neural Networks/test.csv', usecols=range(1,15), skiprows=1, header=None)

    y = np.array(X_train.iloc[:, 14])
    X_train = np.array(X_train.iloc[:, :14])
    X_test = np.array(X_test)

    X_train = NeuralNetTorch.setContiniousValue(X_train)
    X_test = NeuralNetTorch.setContiniousValue(X_test)

    # y_test = np.array(X_test.iloc[:, 4])
    # X_test = np.array(X_test.iloc[:, :4])

    # X_train = X_train.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'O' else col)
    # X_test = X_test.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'O' else col)

    for i in range(X_train.shape[1]):
        if np.issubdtype(X_train[:, i].dtype, object):
            X_train[:, i] = LabelEncoder().fit_transform(X_train[:, i])

    for i in range(X_test.shape[1]):
        if np.issubdtype(X_test[:, i].dtype, object):
            X_test[:, i] = LabelEncoder().fit_transform(X_test[:, i])

    X_train = X_train.astype(np.float32) 
    X_test = X_test.astype(np.float32) 

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y).float()

    X_test = torch.from_numpy(X_test).float()
    # y_test = torch.from_numpy(y_test).float()

    dataset = TensorDataset(X_train, y_train)

    train_dataset = DataLoader(dataset, batch_size=10, shuffle=True)

    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]

    print("RELU:\n")

    count1 = 0
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
            # predicted_train = nnmodel.predict(X_train)
            predicted_test = nnmodel.predict(X_test)

            fileName = f'myfile{count1}.csv'
            # print("Training error: ", nnmodel.calculateError(y_train, predicted_train))
            predicted_test = predicted_test.reshape(len(predicted_test), )
            with open(fileName, "w") as w:
                w.write("ID,Prediction\n")
                id = 1
                for prediction in predicted_test:
                    # prediction = TestPredict(line, rootNode, np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]))
                    w.write(f'{id},{prediction}\n')
                    id += 1
            # print("Test error: ", nnmodel.calculateError(y_test, predicted_test))
            count1 += 1

    print("\nTANH:\n")
    count2 = 0
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
            # predicted_train = nnmodel.predict(X_train)
            predicted_test = nnmodel.predict(X_test)

            fileName = f'myfileT{count2}.csv'
            # print("Training error: ", nnmodel.calculateError(y_train, predicted_train))
            predicted_test = predicted_test.reshape(len(predicted_test), )
            with open(fileName, "w") as w:
                w.write("ID,Prediction\n")
                id = 1
                for prediction in predicted_test:
                    # prediction = TestPredict(line, rootNode, np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]))
                    w.write(f'{id},{prediction}\n')
                    id += 1
            # print("Test error: ", nnmodel.calculateError(y_test, predicted_test))
            count2 += 1