import numpy as np
import pandas as pd

def GetNeibour(idx, border, length):
    rang = border / 2
    idx1 = idx - rang
    idx2 = idx + rang
    if idx1 < 0:
        idx1 = 0
        idx2 = 150
    if idx2 > length:
        idx1 = length - 150
        idx2 = length
    return np.arange(idx1, idx2).astype(np.int)

file1 = pd.read_csv(r'/Users/duanzy/Desktop/code1/2-detection/data/p.csv')
file2 = pd.read_csv(r'/Users/duanzy/Desktop/code1/2-detection/data/p1.csv')
data1, data2 = np.array(file1)[:, 1], np.array(file2)[:, 1]
label1, label2 = np.array(file1)[:, 2], np.array(file2)[:, 2]
Data = []
Label = []
border = 150


for idx, item in enumerate(data1):
    Data.append(data1[GetNeibour(idx=idx, border=150, length=data1.shape[0])])
    Label.append(label1[idx])

Data, Label = np.array(Data), np.array(Label)
np.savez(r'/Users/duanzy/Desktop/code1/2-detection/p', data=Data, label=Label)

'''
for idx, item in enumerate(data2):
    Data.append(data2[GetNeibour(idx=idx, border=150, length=data2.shape[0])])
    Label.append(label2[idx])

Data, Label = np.array(Data), np.array(Label)
np.savez(r'/Users/duanzy/Desktop/code1/2-detection/data2', data=Data, label=Label)
'''

