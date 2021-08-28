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
pred_miss = pd.read_csv("Unet_output.csv",names=['time','f1','f2','f3','f4','f5','f6','f7','f8'],index_col='time')

pred_miss = np.array(pred_miss)[1:,:]
target_data = np.array(ideal_input[int(len(ideal_input)*0.2):int(len(ideal_input)*0.4)])
#convert the train and target set in shape of (batch,sequence length,input_dimension)
train_set = pred_miss[0:int(len(pred_miss)/n_t)*n_t].reshape(int(len(pred_miss)/n_t),n_t,8)
target_train = target_data[0:int(len(target_data)/n_t)*n_t].reshape(int(len(target_data)/n_t),n_t,8)
#create the test set
test_set = np.array(real_input[int(len(real_input)*0.4):int(len(real_input)*0.6)])
target_test = np.array(ideal_input[int(len(ideal_input)*0.4):int(len(ideal_input)*0.6)])
#convert the test and target set in shape of (batch,sequence length,input_dimension)
test_set = test_set[0:int(len(test_set)/n_t)*n_t].reshape(int(len(test_set)/n_t),n_t,8)
target_test = target_test[0:int(len(target_test)/n_t)*n_t].reshape(int(len(target_test)/n_t),n_t,8)
#load the set into Dataset
class JamData(data.Dataset):
    def __init__(self,inputs,targets):
        super(JamData,self).__init__()
        self.inputs = inputs
        self.targets = targets
        
        self.inputs_dtype = torch.float
        self.targets_dtype = torch.float

    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,index:int):
        #select the sample
        x = self.inputs[index]
        y = self.targets[index]
        
        #typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        return x,y

inputs = []
targets =[]
for i in range(len(train_set)):
    inputs.append(train_set[i])
    targets.append(target_train[i])

training_dataset = JamData(inputs,targets)
example, label = training_dataset[1]
training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=4,shuffle=True)
testing_dataset = JamData(test_set,target_test)
testing_dataloader = data.DataLoader(dataset=testing_dataset, batch_size=2,shuffle=False)
print(label.shape)
x, y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

bs = 1
seq_len = 32
input_dim = 8
hidden_dim = 100
n_layers = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

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
        predictions = self.linear(lstm_out)
        return predictions
    
model = LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0
    i = 0
    for inputs, targets in training_dataloader:
        #Move the batch to the device we are using. 
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()# ∇_Θ just got computed by this one call!
        optimizer.step()
        running_loss += loss.item()
        targets = targets.detach().cpu()
        outputs = outputs.detach().cpu()
        #delete the variables to free the memory
        del targets
        del outputs
        if i % 20 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            
            running_loss = 0.0
            
        
        i+=1
        torch.cuda.empty_cache()
print('Finished Training lstm')
PATH = './lstm.pth'
torch.save(model.state_dict(), PATH)
# #Define the U-Net model
class Encoder_block(tf.keras.Model):
    def __init__(self,filters=64,kernel_size=5,**kwargs):
        super(Encoder_block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv2 = tf.keras.layers.Conv2D(self.filters,self.kernel_size,strides=(1,1),padding='SAME')
        self.ac3 = tf.keras.layers.Activation('relu')
        self.conv4 = tf.keras.layers.Conv2D(self.filters,self.kernel_size,strides=(1,1),padding='SAME')
        self.ac5 = tf.keras.layers.Activation('relu')
        
    def call(self, inputs):
        x = self.conv2(inputs)
        x = self.ac3(x)
        x = self.conv4(x)
        x = self.ac5(x)
        
        return x

class Unet_trial(tf.keras.Model):
    def __init__(self,start_filters,**kwargs):
        super(Unet_trial,self).__init__()
        self.start_filters = start_filters
        self.block1 = Encoder_block(self.start_filters*1,5)
        self.block2 = Encoder_block(self.start_filters*2,5)
        self.block3 = Encoder_block(self.start_filters*4,5)
        self.block4 = Encoder_block(self.start_filters*8,5)
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='valid')
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='valid')
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='valid')
        self.up_sampling1 = tf.keras.layers.UpSampling2D(size=(2,2))
        self.up_sampling2 = tf.keras.layers.UpSampling2D(size=(2,2))
        self.up_sampling3 = tf.keras.layers.UpSampling2D(size=(2,2))
        self.up_block1 = Encoder_block(self.start_filters*4,5)
        self.up_block2 = Encoder_block(self.start_filters*2,5)
        self.up_block3 = Encoder_block(self.start_filters*1,5)
        self.up_last = tf.keras.layers.Conv2D(1,1,strides=(1,1),padding='SAME',activation='relu')
        
    def call(self, inputs):
        down_conv1 = self.block1(inputs)
        down_sampling1 = self.max_pool(down_conv1)
        down_conv2 = self.block2(down_sampling1)
        down_sampling2 = self.max_pool1(down_conv2)
        down_conv3 = self.block3(down_sampling2)
        down_sampling3 =self.max_pool2(down_conv3)
        down_conv4 = self.block4(down_sampling3)
        
        up_sampling1 = self.up_sampling1(down_conv4)
        concat1 = concatenate([up_sampling1,down_conv3])
        
        up_conv1 = self.up_block1(concat1)
        
        up_sampling2 = self.up_sampling2(up_conv1)
        concat2 = concatenate([up_sampling2,down_conv2])
        
        up_conv4 = self.up_block2(concat2)
        up_sampling3 = self.up_sampling3(up_conv4)
        concat3 = concatenate([up_sampling3,down_conv1])
        
        up_conv8 = self.up_block3(concat3)
        output_layer = self.up_last(up_conv8)

        return output_layer
Unet = Unet_trial().cuda()
Unet.load_state_dict(torch.load('Pytorch Unet/unet_layer1_data100.pth'))
Unet.eval()
model.eval()
output_lstm = torch.zeros((64,8))
#disable grad
with torch.no_grad():
    for inputs, targets in testing_dataloader:
        inputs = inputs.to(device)
        outputs1 = Unet(inputs)
        outputs = model(outputs1.squeeze())
        outputs = outputs.detach().cpu().squeeze()
        outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[2])
        #print(outputs.shape)
        output_lstm=torch.cat((output_lstm,outputs),axis=0)
        torch.cuda.empty_cache()

pd.DataFrame(output_lstm.numpy()[64:]).to_csv("Pytorch Unet/lstm_output_pytorch.csv")





    

  
