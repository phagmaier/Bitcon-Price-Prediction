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




df = pd.read_csv('test_data.csv')
#print(len(df.columns))
matrix = df.corr()
plt.figure()
sns.heatmap(matrix, annot = True)
plt.show()
col = np.array(df.iloc[:,-1])
#col_tensor = torch.from_numpy(col.values)
df = df.drop(df.columns[-1], axis=1)
df = df.drop(df.columns[0], axis=1)
data_np = df.values
variance = np.var(col)
#added new
mm = MinMaxScaler()
scaler = StandardScaler()
mm.fit(data_np)
data_np = mm.transform(data_np)
scaler.fit(data_np)
data_np = scaler.transform(data_np)

features = torch.from_numpy(data_np).type(torch.float32)
labels = torch.from_numpy(col).type(torch.float32)

train_size = int(0.8 * len(features))

X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=train_size)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)





"""

learning_rate = 0.001
torch.manual_seed(42)
#model_0 = StockPred(13,(13*16*2),(13*16*4), (13*16),1,.2)
model_0 = StockPred(13,(13*32),(13*32*2), (13*16),1,.2)
#input_size, hidden_size, num_layers, output_size
#model_0 = LSTMNetwork(13,(13*16),3,1)
#model_0 = LSTMNetwork()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model_0.parameters(), lr=learning_rate)

#TRAINING LOOP
#Setting at 2 just for error testing

epochs = 500
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





#NOTE:
#Obviously want loss to be 0 but
#A common practice is to compare the MSE loss of the model with the 
#variance of the target variable, 
#if the MSE is lower than the variance it's usually considered a good result.
#BY THE WAY TAKE VARIANCE OF TATGETS NOT FEATURES
#THIS CAN BE DONE BY: var_X = df['X'].var()
#do this in data frame or with numpy array you can do: 
#variance = np.var(sample_array)


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
#min_value = min(testing_loss)
print()
print()
testing_loss = np.array(testing_loss)
#print(min_value)
#print(testing_loss.index(min_value))
print(f"average testing loss is: {np.mean(testing_loss)}")

print()

#SAVING THE MODEL

PATH = "model_dict.pt"
#torch.save(model_0.state_dict(), PATH)


#WHAT I WANT TO DO NOW IS PLOT MY PREDICTIONS AND COMPARE IT TO THE ACTUAL VALUES
plt.figure()
sns.scatterplot(x=range(len(labels)), y=labels, label='Labels')
sns.scatterplot(x=range(len(preds)), y=preds, label='Predictions')
plt.legend()
plt.show()

"""
