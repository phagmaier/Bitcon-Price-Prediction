import torch
import torch.nn as nn

"""
class StockPred(nn.Module):
    def __init__(self,input_size,output_size,shrink_size=None):
        super(StockPred,self).__init__()
        if shrink_size == None:
            shrink_size = output_size
        self.layer_1 = nn.Linear(input_size,output_size)
        self.layer_2 = nn.Linear(output_size,output_size)
        self.layer_3 = nn.Linear(output_size,shrink_size)
        self.layer_4 = nn.Linear(shrink_size, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        #x = F.relu(self.layer_1(x))
        #x = F.relu(self.layer_2(x))
        #x = F.relu(self.layer_3(x))
        #x = self.layer_4(x)
        #return x
        #return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
        return self.layer_4(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))

"""

"""
class StockPred(nn.Module):
    def __init__(self,input_size,hidden_size_1,hidden_size_2,shrink_size,output_size, dropout=0.5):
        super(StockPred,self).__init__()
        self.layer_1 = nn.Linear(input_size,hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1,hidden_size_2)
        self.layer_3 = nn.Linear(hidden_size_2,shrink_size)
        self.layer_4 = nn.Linear(shrink_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer_3(x))
        x = self.dropout(x)
        x = self.layer_4(x)
        return x

"""

class StockPred(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, shrink_size, output_size, dropout=0.5):
        super(StockPred, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_3 = nn.Linear(hidden_size_2, shrink_size)
        self.layer_4 = nn.Linear(shrink_size, output_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tanh(self.layer_1(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_2(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_3(x))
        x = self.dropout(x)
        x = self.layer_4(x)
        return x


"""
class StockPred(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, shrink_size, output_size, dropout=0.5):
        super(StockPred, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_3 = nn.Linear(hidden_size_2, shrink_size)
        self.layer_4 = nn.Linear(shrink_size, hidden_size_2)
        self.layer_5 = nn.Linear(hidden_size_2, hidden_size_1)
        self.layer_6 = nn.Linear(hidden_size_1, output_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tanh(self.layer_1(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_2(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_3(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_4(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_5(x))
        x = self.dropout(x)
        x = self.layer_6(x)
        return x
"""

