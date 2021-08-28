#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:46:30 2020

@author: Phuonglun
"""
import collections
from collections import deque

from typing import Dict, Any, OrderedDict, Tuple, List

import gym
from gym import spaces
import tensorflow as tf
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class JammerEnv(gym.Env):
    
    FREQ_SWITCH_DELAY = 1  # number of subframes need to hop to another frequency
    number_historic_time = 10  #number of previous time slots to present the spectrum image
    
    
    
    def __init__(self, n_jam_freqs,n_com_freqs, n_t,sweep_rate,freq_resolution):
    
        super(JammerEnv, self).__init__()
        
        self.n_jam_freqs = n_jam_freqs
        self.n_t = n_t #number of previous time slots to present the spectrum image
        self.sweep_rate = sweep_rate #KHz/ms
        self.freq_resolution = freq_resolution
        self.action_spaces = spaces.Discrete(n_com_freqs)
        self.num_sweep_freqs = int(self.sweep_rate/self.freq_resolution)
        
        self.whole_observation_spaces = spaces.MultiBinary([n_t,n_jam_freqs])
        self.current_obs_space =  spaces.MultiBinary(n_jam_freqs)
        
        self.reward_range = (0.0, 1.0)
        
    
    
    def reset_comb(self):     
        #self.t = 0
        #init_current_obs  has shape array(n_jam_freqs,)
        self.init_current_obs = self.current_obs_space.sample()
        
        #if type_jam=="comb":
            
        self.init_obs = np.tile(self.init_current_obs,self.n_t).reshape(self.n_t,self.n_jam_freqs,1)
            
        return self.init_obs
    
    def reset_sweep(self,jam_freq_array):
        self.init_current_obs = np.zeros(self.n_jam_freqs)
        for i in range(len(jam_freq_array)):
            self.init_current_obs[jam_freq_array[i]]=1
        temp_init = self.init_current_obs
        init_obs_array = temp_init
        for _ in range(self.n_t-1):
            shift_left_temp_init = np.roll(temp_init,-self.num_sweep_freqs)
            init_obs_array = np.hstack((init_obs_array,shift_left_temp_init))
            temp_init = shift_left_temp_init
        self.init_obs = init_obs_array.reshape(self.n_t,self.n_jam_freqs,1)
        return self.init_obs
        # elif type_jam=="sweep_2":
        #     temp_init = self.init_current_obs
        #     init_obs_array = temp_init
        #     for _ in range(self.n_t-1):
        #         shift_left_temp_init = np.roll(temp_init,-2)
        #         init_obs_array = np.hstack((init_obs_array,shift_left_temp_init))
        #         temp_init = shift_left_temp_init
        #     self.init_obs = init_obs_array.reshape(self.n_t,self.n_jam_freqs,1)
        #     return self.init_obs
    
    def reset_random(self):
        self.init_obs = self.whole_observation_spaces.sample().reshape(self.n_t,self.n_jam_freqs,1)
        return self.init_obs
        
            
    
    def step_comb(self,current_obs):
         #current_obs has shape (n_t, n_jam_freqs,1)
        
            
        self.next_obs = current_obs
            
        return self.next_obs
    
    def step_sweep(self,current_obs):
        
        shift_left_obs = np.roll(current_obs[len(current_obs)-1],-self.num_sweep_freqs)
        added_obs = np.vstack((current_obs,shift_left_obs.reshape(1,len(shift_left_obs),1)))
        self.next_obs = np.delete(added_obs,0,0)
        return self.next_obs
        # elif type_jam=="sweep_2":
        #     shift_left_obs = np.roll(current_obs[len(current_obs)-1],-2)
        #     added_obs = np.vstack((current_obs,shift_left_obs.reshape(1,len(shift_left_obs),1)))
        #     self.next_obs = np.delete(added_obs,0,0)
        #     return self.next_obs
       
        

    def step_random(self,current_obs): 
            
        new_obs = self.current_obs_space.sample().reshape(1,self.n_jam_freqs,1)
        added_obs = np.vstack((current_obs,new_obs))
        self.next_obs = np.delete(added_obs,0,0)
        return self.next_obs
    
    
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
    
    