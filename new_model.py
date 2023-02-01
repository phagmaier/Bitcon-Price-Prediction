import torch
import torch.nn as nn
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