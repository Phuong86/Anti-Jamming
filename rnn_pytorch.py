# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:37:32 2021

@author: tluong
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
torch.cuda.empty_cache()


n_t = 32

'1 jam 100 sweep rate'
real_input = pd.read_csv("CFH_2021_08_09_POS_pulsed/INFERENCE_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col='TIME STEP')
ideal_input = pd.read_csv("CFH_2021_08_09_POS_pulsed/TRAINING_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col =['TIME STEP'])
pred_miss = pd.read_csv("cnn_layer3_toffset_10.csv",names=['time','f1','f2','f3','f4','f5','f6','f7','f8'],index_col='time')


pred_miss = np.array(pred_miss)[1:,:]

pred_train_1jam = pred_miss[0:int(len(pred_miss)*0.5)]
pred_test_1jam = pred_miss[int(len(pred_miss)*0.5):]


target_data = np.array(ideal_input[int(len(ideal_input)*0.8):])

target_train = target_data[0:int(len(target_data)*0.5)]
target_test = target_data[int(len(target_data)*0.5):]


train_set = pred_train_1jam[0:int(len(pred_train_1jam)/n_t)*n_t].reshape(int(len(pred_train_1jam)/n_t),n_t,8)


target_train = target_train[0:int(len(target_train)/n_t)*n_t].reshape(int(len(target_train)/n_t),n_t,8)


test_set = pred_test_1jam[0:int(len(pred_test_1jam)/n_t)*n_t].reshape(int(len(pred_test_1jam)/n_t),n_t,8)
target_test = target_test[0:int(len(target_test)/n_t)*n_t].reshape(int(len(target_test)/n_t),n_t,8)

    


bs = 1
seq_len = 32
input_dim = 8
hidden_dim = 100
n_layers = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

train_set = torch.tensor(train_set, dtype=torch.float).to(device)
target_train = torch.tensor(target_train, dtype=torch.float).to(device)

test_set = torch.tensor(test_set, dtype=torch.float).to(device)
target_test = torch.tensor(test_set, dtype=torch.float).to(device)
class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=100, n_layers =1, output_size=8, seq_len = 32):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_layer_size, n_layers)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        # self.hidden_cell = (torch.randn(n_layers,seq_len,self.hidden_layer_size),
        #                     torch.randn(n_layers,seq_len,self.hidden_layer_size))

    def forward(self, input_seq):
        hid_cell =  torch.zeros(self.n_layers, self.seq_len, self.hidden_layer_size,device=input_seq.device).float()
        state_cell = torch.zeros(self.n_layers, self.seq_len, self.hidden_layer_size,device=input_seq.device).float()
        #lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq, (hid_cell,state_cell))
        predictions = self.linear(lstm_out)#.view(len(input_seq), -1))
        return predictions
    
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer_lstm = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
epochs = 5
running_loss=[]
for i in range(epochs):
    for i in  range(len(train_set)-1):
        optimizer_lstm.zero_grad()
      

        y_pred = model(train_set[i:i+1])

        single_loss = loss_function(y_pred, target_train[i:i+1])
        single_loss.backward()
        optimizer_lstm.step()
        running_loss.append(single_loss.item())

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
print('Finished Training lstm')
PATH = './lstm_10.pth'
torch.save(model.state_dict(), PATH)

output_lstm = model(test_set[0:1])
output_lstm = output_lstm.detach().cpu().numpy()
output = output_lstm.reshape(output_lstm.shape[1],output_lstm.shape[2])

for i in range(1,len(test_set)):
    output_lstm = model(test_set[i:i+1])
    output_lstm = output_lstm.detach().cpu().numpy()
    output_lstm = output_lstm.reshape(output_lstm.shape[1],output_lstm.shape[2])
    output = np.vstack((output,output_lstm))
    
y_binary = output
for ii in range(0,len(y_binary)):
    y_binary[ii]=np.where(y_binary[ii]>0.5,1,y_binary[ii])
    y_binary[ii]=np.where(y_binary[ii]<=0.5,0,y_binary[ii])





    

  
