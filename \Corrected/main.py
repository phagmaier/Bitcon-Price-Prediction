import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sklearn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
from new_model import StockPred

df = pd.read_csv('cleaned_data.csv')

matrix = df.corr()
plt.figure()
sns.heatmap(matrix,annot=True)
#plt.show()

#get labels in an array
labels = np.array(df.iloc[:,-1])
#calculate variance on labels
variance = np.var(labels)

#drop the labels column
df = df.drop(df.columns[-1], axis=1)
#drop index col that still shows up when i tell pandas not to include it
df = df.drop(df.columns[0], axis=1)

#also gonna drop circsupply
df = df.drop(df.columns[4],axis=1)
df = df.drop(df.columns[6],axis=1)


#plt.show()
#going to drop some other values

#get features
data_np = df.values


#scale the data using a min max scaler and standard

mm = MinMaxScaler()
scaler = StandardScaler()
mm.fit(data_np)
data_np = mm.transform(data_np)
scaler.fit(data_np)
data_np = scaler.transform(data_np)


features = torch.from_numpy(data_np).type(torch.float32)
labels = torch.from_numpy(labels).type(torch.float32)

train_size = int(0.8 * len(features))

np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=train_size)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


#With this model: model_0 = StockPred(10,(10*16),(10*32), (10*16),1,.2)

#LR 50: 224545539
#LR 20: 203136238
#LR 10: 88888636
#LR 05: 65393688
#LR 01: 231920351
#LR .1: 105235401

#NEW MODEL: StockPred(10,(10*32),(10*32), (10),1,.2)
#LR 05: 109336971
#LR .1: 59075918
#L .01: 293435514


#EVEN LESS LAYERS: StockPred(10,(10),(10), (5),1,.2)
#: .001: 688065516
#L: .01: 675136635
#L: 0.1: 418418491
#L: 001: 235490371
#L: 005: 258590999
#L: 020: 241205221

#THE SMALLEST: StockPred(10,(8),(6), (3),1,.2)
#LR: 005: 172039992
#LR: 001: 172039992
#LR: .01: 671508982
#LR: 0.1: 460044819
#L:.0001: 712944219


#235221760 -> .001
#433329264
#131225543
#61699877
#443422685
#108283251
#87436438
#72270933
#88053426
#68248350
#428398688

"""
learning_rate = 1
torch.manual_seed(42)
#before multiply by 2: 70432455
model_0 = StockPred(10,(10*16),(10*16), (10),1,.2)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model_0.parameters(), lr=learning_rate)


#TRAINING
epochs = 200
for epoch in range(epochs):
    train_loss = []
    model_0.train()
    for X,y in train_dataloader:
        optim.zero_grad()
        output = model_0(X)
        y = y.view(-1,1)
        loss = loss_fn(output, y)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
    loss_now = np.mean(train_loss)
    if epoch % 10 == 0:
        print(f"Loss for epoch {epoch} : {loss_now}")



#TESTING:
testing_loss = []
preds = np.array([])
labels = np.array([])
with torch.inference_mode():
    for X,y in test_dataloader:
        labels = np.append(labels,[y.item()])
        output = model_0(X)
        preds = np.append(preds,[output.item()])
        y = y.view(-1,1)
        loss = loss_fn(output,y)
        testing_loss.append(loss.item())
print()
print()
testing_loss = np.array(testing_loss)
print(f"average testing loss is: {np.mean(testing_loss)}")



plt.figure()
sns.scatterplot(x=range(len(labels)), y=labels, label='Labels')
sns.scatterplot(x=range(len(preds)), y=preds, label='Predictions')
plt.legend()
print(variance)
plt.show()
"""


#Should try this model with 8 features
#TRYING WITH ONLY 8 features
#1319841
#923345
#902194 -> model_0 = StockPred(8,(8*64*8),(1),0.2) on .05
#981344
#755273 ->model_0 = StockPred(8,(8*64*8*8),(1),0.2) on .05
#730047 ->model_0 = StockPred(8,(8*64*8*8),(1),0.2) on .01
#1319841
#902194
#981344
learning_rate = .005
torch.manual_seed(42)
#before multiply by 2: 70432455
model_0 = StockPred(8,(64*64*8),(1),0.2)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model_0.parameters(), lr=learning_rate)


#TRAINING
epochs = 200
for epoch in range(epochs):
    train_loss = []
    model_0.train()
    for X,y in train_dataloader:
        optim.zero_grad()
        output = model_0(X)
        y = y.view(-1,1)
        loss = loss_fn(output, y)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
    loss_now = np.mean(train_loss)
    if epoch % 10 == 0:
        print(f"Loss for epoch {epoch} : {loss_now}")





#TESTING:
testing_loss = []
preds = np.array([])
labels = np.array([])
with torch.inference_mode():
    for X,y in test_dataloader:
        labels = np.append(labels,[y.item()])
        output = model_0(X)
        preds = np.append(preds,[output.item()])
        y = y.view(-1,1)
        loss = loss_fn(output,y)
        testing_loss.append(loss.item())
print()
print()
testing_loss = np.array(testing_loss)
print(f"average testing loss is: {np.mean(testing_loss)}")



plt.figure()
sns.scatterplot(x=range(len(labels)), y=labels, label='Labels')
sns.scatterplot(x=range(len(preds)), y=preds, label='Predictions')
plt.legend()
print(variance)
plt.show()
