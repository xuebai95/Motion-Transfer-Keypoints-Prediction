import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,batch_size = 250):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.gru(x, h0.detach())
        out = self.fc(out[:, :, :]) # 5-5,15-15
#         out = self.fc(out[:, -5:, :]) # 10-5

        return out    
        
# 5-10 frames
class GRUModel_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size=250):
        super(GRUModel_2, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, future=0):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.gru(x, h0.detach())
        out = self.fc(out[:, :, :]) # 6 steps forward
        
        if future > 0:
            # Predict 'future' time steps
            for _ in range(future):
                # Use the last predicted output as the input for the next time step
                last_output = out[:, -5:, :]
#                 print(last_output.shape)
                out_pred, _ = self.gru(last_output.detach(), h0.detach())
                out_pred = self.fc(out_pred[:, -5:, :])
#                 print(out_pred.shape)
                out = torch.cat([out, out_pred], dim=1)

        

        return out