import torch as t
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=2, bias=False),
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(4),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 6, kernel_size=2, bias=False), 
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(6),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(6, 8, kernel_size=2, bias=False), 
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(8),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(136, 24),
            nn.ReLU(),
            nn.Linear(24, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MyDataset(Dataset):
    def __init__(self, DataArray, LabelArray):
        super(MyDataset, self).__init__()
        self.data = DataArray
        self.label = LabelArray

    def __getitem__(self, index):
        return t.tensor(self.data[index], dtype=t.float32).cpu(), t.tensor(self.label[index], dtype=t.long).cpu()

    def __len__(self):
        return self.label.shape[0]

def train(net, dataloader, testdataloader, optimizer, criterion, epocs=10):
    for epoc in range(epocs):
        net.train()
        TrainLoss = 0
        Correct = 0
        Total = 0
        for BatchIdx, (InputData, Labels) in enumerate(dataloader):
            Outputs = net(InputData)
            optimizer.zero_grad()
            loss = criterion(Outputs, Labels)
            loss.backward()
            optimizer.step()
            _, predicted = Outputs.max(1)
            Total = Total + Outputs.size(0)
            Correct = Correct + predicted.eq(Labels).sum().item()

            if BatchIdx % 100 == 0 and BatchIdx > 0:
                TrainLoss = loss.item()
                print('Batch:\t', BatchIdx + 1, '/', len(dataloader), '\t\tLoss:\t', round(TrainLoss, 2), 
                    '\t\tAccuracy:\t', round(Correct/Total, 3), '\t(', Correct, '/', Total, ')')
        TestLoss = 0
        Correct = 0
        Total = 0
        with t.no_grad():
            for BatchIdx, (InputData, Labels) in enumerate(testdataloader):
                Outputs = net(InputData)
                loss = criterion(Outputs, Labels)
                TestLoss = loss.item()
                _, predicted = Outputs.max(1)
                Total = Total + Outputs.size(0)
                Correct = Correct + predicted.eq(Labels).sum().item()
            print('Epocs:\t', epoc + 1, '/', epocs, '\t\tLoss:\t', round(TestLoss, 3), 
                    '\t\tAccuracy:\t', round(Correct/Total, 3), '\t(', Correct, '/', Total, ')')
            print('-'*60)
                

SplitRatio = 0.9
TrainBatchSize = 16
TestBatchSize = 16
lr = 0.001
criterion = nn.CrossEntropyLoss()
epocs = 250

# arr = np.load('Detection\data1.npz')
arr = np.load('/Users/duanzy/Desktop/code1/2-detection/p1.npz')


Data, Label = arr['data'], arr['label']
Data = np.expand_dims(Data, axis=1)
idx = np.random.permutation(Data.shape[0])
SplitPoint = int(Data.shape[0] * SplitRatio)
TrainData, TestData = Data[idx][:SplitPoint], Data[idx][SplitPoint:]
TrainLabel, TestLabel = Label[idx][:SplitPoint], Label[idx][SplitPoint:]
TrainSet = MyDataset(TrainData, TrainLabel)
TestSet = MyDataset(TestData, TestLabel)
TrainDataLoader = DataLoader(dataset=TrainSet, batch_size=TrainBatchSize, shuffle=True)
TestDataLoader = DataLoader(dataset=TestSet, batch_size=TestBatchSize, shuffle=True)

net = CNN().cpu()
optimizer = t.optim.Adam(net.parameters(), lr = lr)
train(net, TrainDataLoader, TestDataLoader, optimizer, criterion, epocs=epocs)

Data = t.tensor(Data, dtype=t.float32).cpu()
res = net(Data).argmax(axis=1).cpu().numpy()
print('Precision:{}'.format(precision_score(Label, res)))
print('Recall:{}'.format(recall_score(Label, res)))
print('F1-score:{}'.format(f1_score(Label, res)))
x = []
for idx in range(Data.shape[0]):
    if res[idx] == 1:
        x.append(idx)
np.savetxt('p1.txt', np.array(x).reshape(-1, 1).astype(np.int), fmt='%s')
