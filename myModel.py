import torch
import torch.nn as nn
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