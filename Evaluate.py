#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:32:42 2021

@author: Phuonglun
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import matplotlib.pyplot as plt
from Jamming_env import JammerEnv
import random
import pandas as pd
#tf.enable_eager_execution()

new_data = pd.read_csv('phuong_variable_sweep_25_jam_1/PHUONG_2_FL_PHUONG_PARAM_RANGE_11_REGION_01_PARAM_01_01_DATA.csv')
f1_data=new_data['f 1']
f2_data = new_data['f 2']
f3_data = new_data['f 3']
f4_data = new_data['f 4']
f5_data = new_data['f 5']
f6_data = new_data['f 6']
f7_data = new_data['f 7']
f8_data = new_data['f 8']
f9_data = new_data['f 9']
f10_data = new_data['f10']



n_com_freqs = 10

n_outputs = 10
n_jam_freqs = 10
n_t = 10

tf.reset_default_graph()

def q_network(X,name_scope):
    #initialize layers
    initializer = tf.contrib.layers.variance_scaling_initializer()
    
    with tf.variable_scope(name_scope) as scope:
        
        # initialize the convolutional layers
        layer_1 = conv2d(X, num_outputs=16, kernel_size=(8,8), stride=4, padding='SAME', weights_initializer=initializer) 
        tf.summary.histogram('layer_1',layer_1)
        
        layer_2 = conv2d(layer_1, num_outputs=32, kernel_size=(4,4), stride=2, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_2',layer_2)
        
        flat = flatten(layer_2)
        
        #feed input state to fully connected layer
        fc_1 = fully_connected(flat, num_outputs=256, weights_initializer=initializer)
        tf.summary.histogram('fc_1',fc_1)
        
        fc_2 = fully_connected(fc_1, num_outputs=32, weights_initializer=initializer)
        tf.summary.histogram('fc_2',fc_2)
        
        output = fully_connected(fc_2, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.summary.histogram('output',output)
        
        #Vars will store the parameters of the network such as weights 
        
        vars = {v.name[len(scope.name):]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
        return vars, output
    
def reward_func(obs,action):
    #action is array(1,) point the index of 
    shape_o = obs.shape
    current_o = obs[shape_o[0]-1,:].reshape(n_jam_freqs)
    
    if current_o[action]==-101:
        r = 1
    else:
        r=0
    return r

X_shape = (None,n_t,n_jam_freqs,1)
in_training_mode = tf.placeholder(tf.bool)
num_user_switch_freqs = 1 #2ms
X = tf.placeholder(tf.float32, shape=X_shape)
mainQ, mainQ_outputs = q_network(X,'mainQ')    
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'models/Jammer_model_1of10_jammed_variable_rate_period_25.ckpt')

    num_epoch = 499980
    global_step_test = 0
    timestep_reward_test = []
    actions_counter_test = Counter()
    split_data = 500*1000

    obs_test_ = np.vstack((f1_data[split_data:split_data+n_t],f2_data[split_data:split_data+n_t],f3_data[split_data:split_data+n_t],f4_data[split_data:split_data+n_t],f5_data[split_data:split_data+n_t],f6_data[split_data:split_data+n_t],f7_data[split_data:split_data+n_t],f8_data[split_data:split_data+n_t],f9_data[split_data:split_data+n_t],f10_data[split_data:split_data+n_t]))
    obs_test = obs_test_.transpose(1,0).reshape(n_t,n_jam_freqs,1)
    #noise  = np.random.normal(mean_noise,np.sqrt(noise_var),obs.shape)
    #obs_test_noise = obs_test + noise
    
    total_reward_test = 0
    epoch_test = 0
    
    reward_store_test=[]
    for i in range(num_epoch):
    
        #print('current observation', obs_test.reshape(n_t,n_jam_freqs))
        actions_test = mainQ_outputs.eval(feed_dict={X:[obs_test], in_training_mode:False})
        #print("action_output_value:",actions_test)    
            #get action
        action_test = np.argmax(actions_test,axis=-1)
        #print("action:",action_test)
        #action_test=action_test[0]
        #            print(f"argmax,{action}")
        actions_counter_test[str(action_test)]+=1
            
        #action_test = epsilon_greedy(action_test,global_step_test)    
        
           
        #next_obs_test = obs
        #next_obs_test = gym_env.step_sweep(obs_test)
        
        new_line_test = (i+1)*num_user_switch_freqs
        next_index_test_1 = split_data+new_line_test
        next_index_test_2 = split_data+new_line_test+n_t
        next_obs_test_ = np.vstack((f1_data[next_index_test_1:next_index_test_2],f2_data[next_index_test_1:next_index_test_2],f3_data[next_index_test_1:next_index_test_2],f4_data[next_index_test_1:next_index_test_2],f5_data[next_index_test_1:next_index_test_2],f6_data[next_index_test_1:next_index_test_2],f7_data[next_index_test_1:next_index_test_2],f8_data[next_index_test_1:next_index_test_2],f9_data[next_index_test_1:next_index_test_2],f10_data[next_index_test_1:next_index_test_2]))
        #print("next observation:",next_obs_test_.transpose(1,0))
        next_obs_test = next_obs_test_.transpose(1,0).reshape(n_t,n_jam_freqs,1)
        
        """calculate the reward"""
        
        reward_test = 0
        
        reward_test += reward_func(next_obs_test,action_test)
        reward_store_test.append(reward_test)

        
                
        total_reward_test += reward_test 
        obs_test = next_obs_test
        #noise  = np.random.normal(mean_noise,np.sqrt(noise_var),obs.shape)
        #obs_test_noise = obs_test + noise
        global_step_test +=1
        epoch_test += 1
        
        timestep_reward_test.append(total_reward_test)
        print('Epoch test', epoch_test, 'Reward test', reward_test,) 


#plt.plot(episode_reward)
print('accuracy is',np.mean(reward_store_test))