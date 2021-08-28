# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:27:30 2021

@author: tluong
"""

import tensorflow as tf
#tf.config.run_functions_eagerly(True)
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
import numpy as np


n_len = 32
'1jammed  100 sweep rate'
real_input = pd.read_csv("CFH_2021_08_09_POS_pulsed/INFERENCE_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col='TIME STEP')
ideal_input = pd.read_csv("CFH_2021_08_09_POS_pulsed/TRAINING_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col =['TIME STEP'])

real_input = real_input.to_numpy()
ideal_input = ideal_input.to_numpy()
real_input = np.where(real_input>1, 0.5, real_input)

train_input_set = real_input[:int(len(real_input)*0.2)]
y_train_set = ideal_input[:int(len(ideal_input)*0.2)]
         
# test_input_set = real_input[int(len(real_input)*0.5):]
# y_test_set = ideal_input[int(len(ideal_input)*0.5):]

train_images = train_input_set.reshape(int(len(train_input_set)/n_len),n_len,8,1)
y_train_images = y_train_set.reshape(int(len(y_train_set)/n_len),n_len,8,1)

# test_images = test_input_set.reshape(int(len(test_input_set)/n_len),n_len,8,1)
# y_test_images = y_test_set.reshape(int(len(y_test_set)/n_len),n_len,8,1)


'1 jammed 250 frame sweep'
# real_input_250sweep = pd.read_csv("CFH_2021_08_09_POS_pulsed/INFERENCE_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col='TIME STEP')
# ideal_input_250sweep = pd.read_csv("CFH_2021_08_09_POS_pulsed/TRAINING_1_jammed_freq_sweep_rate_10_PULSED_4_6.txt",usecols=['TIME STEP',' f1','f2','f3','f4','f5','f6','f7','f8'],index_col =['TIME STEP'])

# real_input_250sweep = real_input_250sweep.to_numpy()
# ideal_input_250sweep = ideal_input_250sweep.to_numpy()
# real_input_250sweep = np.where(real_input_250sweep>1, 0.5, real_input_250sweep)

# train_input_set_250sweep = real_input_250sweep[:int(len(real_input_250sweep)*0.2)]
# y_train_set_250sweep = ideal_input_250sweep[:int(len(ideal_input_250sweep)*0.2)]
         
# test_input_set_250sweep = real_input_250sweep[int(len(real_input_250sweep)*0.8):]
# y_test_set_250sweep = ideal_input_250sweep[int(len(ideal_input_250sweep)*0.8):]

# train_images_250sweep = train_input_set_250sweep.reshape(int(len(train_input_set_250sweep)/n_len),n_len,8,1)
# y_train_images_250sweep = y_train_set_250sweep.reshape(int(len(y_train_set_250sweep)/n_len),n_len,8,1)

# test_images_250sweep = test_input_set_250sweep.reshape(int(len(test_input_set_250sweep)/n_len),n_len,8,1)
# y_test_images_250sweep = y_test_set_250sweep.reshape(int(len(y_test_set_250sweep)/n_len),n_len,8,1)



# train_input_set = real_input_250sweep[int(len(real_input_250sweep)*0.3):int(len(real_input_250sweep)*0.5)]
# y_train_set = ideal_input_250sweep[int(len(ideal_input_250sweep)*0.3):int(len(ideal_input_250sweep)*0.5)]
# train_images = train_input_set.reshape(int(len(train_input_set)/n_len),n_len,8,1)
# y_train_images = y_train_set.reshape(int(len(y_train_set)/n_len),n_len,8,1)

# test_input_set = real_input_250sweep[int(len(real_input_250sweep)*0.5):int(len(real_input_250sweep)*0.7)]
# y_test_set = ideal_input_250sweep[int(len(ideal_input_250sweep)*0.5):int(len(ideal_input_250sweep)*0.7)]
# test_images = test_input_set.reshape(int(len(test_input_set)/n_len),n_len,8,1)
# y_test_images = y_test_set.reshape(int(len(y_test_set)/n_len),n_len,8,1)



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
    
input = Input(shape=(32,8,1))

Unet = Unet_trial(start_filters=64)
Unet.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

"trainning Unet layer 1"
history= Unet.fit(train_images,y_train_images,epochs=5,batch_size=12, validation_split=0.2)#,callbacks=[earlystopper, checkpointer])
Unet.save('Unet1_data10')

