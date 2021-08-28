#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:05:43 2020

@author: Phuonglun
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:54:59 2020

@author: Phuonglun
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import matplotlib.pyplot as plt
#from Jamming_env import JammerEnv
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

sweep_rate_min = 5
sweep_rate_max = 15
#freq_resolution = 100  #100KHz

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
    
epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000

def epsilon_greedy(action,step):

    epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action 

def reward_func(obs,action):
    #action is array(1,) point the index of 
    shape_o = obs.shape
    current_o = obs[shape_o[0]-1,:].reshape(n_jam_freqs)
    
    if current_o[action]==-101:
        r = 1
    else:
        r=0
    return r

#define replay buffer with length 20000 to hold the experience 
buffer_len = 2000
exp_buffer = deque(maxlen=buffer_len)


#define function to sample the experiences from the memory
def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3]


#define network hyperparameters
num_episodes = 500
batch_size = 48
input_shape = (None,n_t,n_jam_freqs,1)
learning_rate = 0.001
X_shape = (None,n_t,n_jam_freqs,1)
discount_factor = 0.97

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000
max_steps = 1000

 #Now we define the placeholder for our input i.e game state
X = tf.placeholder(tf.float32, shape=X_shape)

# we define a boolean called in_training_model to toggle the training
in_training_mode = tf.placeholder(tf.bool)

# we build our Q network, which takes the input X and generates Q values for all the actions in the state
mainQ, mainQ_outputs = q_network(X, 'mainQ')

# similarly we build our target Q network
targetQ, targetQ_outputs = q_network(X, 'targetQ')

# define the placeholder for our action values
X_action = tf.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(targetQ_outputs * tf.one_hot(X_action, n_outputs), axis=-1, keep_dims=True)

copy_op = [tf.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)

# define a placeholder for our output i.e action
y = tf.placeholder(tf.float32, shape=(None,1))

# now we calculate the loss which is the difference between actual value and predicted value
loss = tf.reduce_mean(tf.square(y - Q_action))

# we use adam optimizer for minimizing the loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
#sweep_rate =600 #600Khz/ms
num_user_switch_freqs = 1 #2ms

with tf.Session() as sess:
    init.run()
    
    # for each episode
    episode_reward = []
    """in the case we do not have the jamming data, we can use gym_env to get the initial observation"""
    #gym_env = JammerEnv(n_jam_freqs,n_com_freqs, n_t, sweep_rate, freq_resolution) 
    #obs = gym_env.reset_sweep()
    """here we load the sequence of jamming data from f1 to f10
    obs_init = np.vstack((f1_data[0:n_t],f2_data[0:n_t],f3_data[0:n_t],f4_data[0:n_t],f5_data[0:n_t],f6_data[0:n_t],f7_data[0:n_t],f8_data[0:n_t],f9_data[0:n_t],f10_data[0:n_t]))
    obs = obs_init.transpose(1,0).reshape(n_t,n_jam_freqs,1)
    
    """we can add noise to the jamming data in practice"""
    #mean_noise = 0
    #noise_var = 1
    
    #noise  = np.random.normal(mean_noise,np.sqrt(noise_var),obs.shape)
    #obs_add_noise = obs + noise
    # obs_noise = obs_add_noise.reshape(n_t,n_jam_freqs,1)
    
    comm_selection =[]    
    
    for i in range(num_episodes):
        done = False
        #obs = env.reset()
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter() 
        episodic_loss = []
        timeslots_reward=[]
        timeslot_reward = 0
            
        t = 0 
        
        total_reward = 0
        while t < max_steps:
            t +=1
        
            #actions = mainQ_outputs.eval(feed_dict={X:[obs_add_noise], in_training_mode:False})
            actions = mainQ_outputs.eval(feed_dict={X:[obs], in_training_mode:False})
            #get action
            action = np.argmax(actions,axis=-1)
            actions_counter[str(action)] += 1 

            # select the action using epsilon greedy policy
            action = epsilon_greedy(action, global_step)
            channels = np.zeros(n_com_freqs) 
            channels[action]=1
            #print("selected channel for communication:",channels)
            comm_selection.append(channels)
            #next observation in random jamming type
            
            #next_o = gym_env.step_sweep(obs)
            if i==0:
                new_line = t*num_user_switch_freqs
                #print('new line:', new_line)
            else:
                new_line = t*num_user_switch_freqs+i*max_steps
                #print('new line:', new_line)
            next_o_ = np.vstack((f1_data[0+new_line:n_t+new_line],f2_data[0+new_line:n_t+new_line],f3_data[0+new_line:n_t+new_line],f4_data[0+new_line:n_t+new_line],f5_data[0+new_line:n_t+new_line],f6_data[0+new_line:n_t+new_line],f7_data[0+new_line:n_t+new_line],f8_data[0+new_line:n_t+new_line],f9_data[0+new_line:n_t+new_line],f10_data[0+new_line:n_t+new_line]))
            next_o = next_o_.transpose(1,0).reshape(n_t,n_jam_freqs,1)
            #next_noise  = np.random.normal(mean_noise,np.sqrt(noise_var),next_o.shape)
    
            #next_obs_add_noise = next_o + next_noise
            
            reward = reward_func(next_o,action)
             
            total_reward += reward
            
            exp_buffer.append([obs,action,next_o,reward])
            #exp_buffer.append([obs_add_noise,action,next_obs_add_noise,reward])
            
            if global_step % steps_train ==0 and global_step > start_steps:
                #sample experience
                o_obs,o_act,o_next_obs,o_reward = sample_memories(batch_size)
                #state
                o_obs = [x for x in o_obs]
                #next state
                o_next_obs = [x for x in o_next_obs]
                
                # next actions
                next_act = mainQ_outputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})
                
                y_batch = o_reward + discount_factor*np.max(next_act,axis=-1)
                train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                episodic_loss.append(train_loss)
                
            if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                copy_target_to_main.run()
            
            obs = next_o
            #noise  = np.random.normal(mean_noise,np.sqrt(noise_var),obs.shape)
            # obs_add_noise = obs + noise
            epoch += 1
            global_step += 1
            #timeslot_reward += reward
            
            episodic_reward += reward
            
            
        print('Episode'.format(i), 'Reward', episodic_reward,)
        episode_reward.append(episodic_reward)
        saver.save(sess,'models/Jammer_model_1of10_jammed_variable_rate_period_25.ckpt')
        
plt.plot(episode_reward)
    



            
            
            
            
            
            
