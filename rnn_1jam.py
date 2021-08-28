
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:31:26 2021

@author: Phuonglun
"""
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import random
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model

n_com_freqs = 8
n_timeslot_ahead = 30

n_jam_freqs =n_com_freqs
n_t = 30

# real_input = pd.read_csv("CFH_2021_08_09_POS_pulsed/INFERENCE_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col='TIME STEP')
# ideal_input = pd.read_csv("CFH_2021_08_09_POS_pulsed/TRAINING_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col =['TIME STEP'])

real_input_30 = pd.read_csv("CFH_2021_08_09_POS_pulsed/INFERENCE_1_jammed_freq_sweep_rate_30_PULSED_12_18.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col='TIME STEP')
ideal_input_30 = pd.read_csv("CFH_2021_08_09_POS_pulsed/TRAINING_1_jammed_freq_sweep_rate_30_PULSED_12_18.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col =['TIME STEP'])


pred_miss = pd.read_csv("cnn_layer3_toffset_30.csv",names=['time','f1','f2','f3','f4','f5','f6','f7','f8'],index_col='time')
'1 jammed 1000 sweep'


pred_miss = np.array(pred_miss)[1:,:]

#pred_miss_1jam = pred_miss[0:int(len(pred_miss)*0.5)]
# pred_train_1jam = pred_miss[0:int(len(pred_miss)*0.5)]
# pred_test_1jam = pred_miss[int(len(pred_miss)*0.5):]


#target_data = np.array(ideal_input[int(len(ideal_input)*0.6):int(len(ideal_input)*0.8)])
target_data_30 = np.array(ideal_input_30[int(len(ideal_input_30)*0.6):int(len(ideal_input_30)*0.8)])

#target_train = target_data[0:int(len(target_data)*0.5)]
#target_test = target_data[int(len(target_data)*0.5):]
#target_data=np.concatenate((target_data,target_data_30),axis=0)

train_set = pred_miss[0:int(len(pred_miss)/n_t)*n_t].reshape(int(len(pred_miss)/n_t),n_t,8)

#target_train = target_data[0:int(len(target_data)/n_t)*n_t].reshape(int(len(target_data)/n_t),n_t,8)
target_train_30 = target_data_30[0:int(len(target_data_30)/n_t)*n_t].reshape(int(len(target_data_30)/n_t),n_t,8)

# test_data = np.array(real_input[int(len(real_input)*0.8):])
test_data_30 = np.array(real_input_30[int(len(real_input_30)*0.8):])

# test_set = test_data[0:int(len(test_data)/n_t)*n_t].reshape(int(len(test_data)/n_t),n_t,8)
test_set_30 = test_data_30[0:int(len(test_data_30)/n_t)*n_t].reshape(int(len(test_data_30)/n_t),n_t,8)

# y_test_data = np.array(ideal_input[int(len(ideal_input)*0.8):])
# y_test_data_30 = np.array(ideal_input_30[int(len(ideal_input_30)*0.8):])

# y_test_set = y_test_data[0:int(len(y_test_data)/n_t)*n_t].reshape(int(len(y_test_data)/n_t),n_t,8)
# y_test_set_30 = y_test_data_30[0:int(len(y_test_data_30)/n_t)*n_t].reshape(int(len(y_test_data_30)/n_t),n_t,8)





num_inputs = 8
num_time_steps = n_t
num_neurons = 50
num_outputs = 8
learning_rate = 0.01
num_iterations = 10
batch_size=32
    

input= Input(shape = (num_time_steps,num_inputs),dtype=tf.float32)

gru = tf.keras.layers.LSTM(num_neurons,activation='tanh',return_sequences=True)(input)
den = tf.keras.layers.Dense(num_outputs,activation='sigmoid')(gru)


model = Model(inputs=input,outputs=den)
#model.compile(optimizer='adam',loss='mse',metrics=['BinaryAccuracy'])
model.compile(optimizer='adam',loss='mse',metrics=[tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives(),'BinaryAccuracy'])
plot_model(model)

history = model.fit(train_set,target_train_30,epochs=20,batch_size=32,validation_split=0.2)
model.save('Lstm_data30')
print(history.history.keys())
#y_pred=model.predict(test_set_30)  
# y_pred_reshape = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])
# pd.DataFrame(y_pred_reshape).to_csv("rnn_output_1jam_float_30.csv")
# # rnn_output = pd.read_csv('rnn_output_1jam_float_10and30.csv')
# # data = rnn_output.to_numpy()
# # y_pred_reshape = data[:,1:9]
# # 'map to binary values [0,1] with threshold =0.5'
# y_binary = y_pred_reshape
# for ii in range(0,len(y_binary)):
#     y_binary[ii]=np.where(y_binary[ii]>0.5,1,y_binary[ii])
#     y_binary[ii]=np.where(y_binary[ii]<=0.5,0,y_binary[ii])
# #y_pred_reshape = y_pred_reshape.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])
# pd.DataFrame(y_binary).to_csv("rnn_output_1jam_binary_30.csv")
# #target_test=np.concatenate((y_test_set,y_test_set_30),axis=0)
# target_test = target_test.reshape(target_test.shape[0]*target_test.shape[1],target_test.shape[2])
# m = tf.keras.metrics.BinaryAccuracy()
# m.update_state(y_binary,target_test)
# print('Binary accuracy in test file:',m.result().numpy())

