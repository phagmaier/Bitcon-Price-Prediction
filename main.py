"""
The following program loads our data from the CSV we generated
We create a heatmap of our original data and we also plot our predictions
compard to the actual labels
What we are trying to predict is the change of price as a percentage since
The epochs in the training can be changed and you can uncoment the 
line that prints the minimum loss value and when it occured in training. Usually
It occurs iaround the 100th epoch but despite this training for anything less than 300 epochs
causes a significant decrease in accuracy on our testing data
"""


import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import sklearn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import classification_report,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
from helper_functions import price_change
from myModel import StockPred


#LOAD IN THE DATA FROM THE CSV
df = pd.read_csv("data.csv")

plt.figure()
sns.heatmap(df, annot = True)
#plt.show()

#Get labels
col = df.iloc[:,0]
col_tensor = torch.from_numpy(col.values)

#get our X data
df = df.drop(df.columns[0], axis=1)
data_np = df.values


col = price_change(col)
variance = np.var(col)


data_np = data_np[:-1]

#normalize data
scaler = StandardScaler()
scaler.fit(data_np)
data_np = scaler.transform(data_np)

data_tensor = torch.from_numpy(data_np)
col = torch.from_numpy(col)
#converting to float 32 for better performance 
labels = col.type(torch.float32)
data = data_tensor.type(torch.float32)

train_size = int(0.8 * len(data))


X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


#MODEL WAS HERE

 


learning_rate = 0.01
torch.manual_seed(42)
model_0 = StockPred(12,98)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model_0.parameters(), lr=learning_rate)

#TRAINING LOOP

epochs = 1000
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
        print(f"Loss for epoch {epoch+1} : {loss_now}")
min_val = min(train_loss)
print()
print()
print(min_val)
print(train_loss.index(min_val))





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
min_value = min(testing_loss)
print()
print()
#print(min_value)
#print(testing_loss.index(min_value))



#SAVING THE MODEL
#PATH = "model_dict.pt"
#torch.save(model_0.state_dict(), PATH)


#Plot results an the heatmap showing the relationships
#between our input features
plt.figure()
sns.scatterplot(x=range(len(labels)), y=labels, label='Labels')
sns.scatterplot(x=range(len(preds)), y=preds, label='Predictions')
plt.legend()
plt.show()
