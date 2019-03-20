# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:57:48 2019

@author: Lucas
"""

import numpy as np
from numpy import random
from linucb_data import dataset_3
import matplotlib.pyplot as plt

# linear bandit (ucb) #

class LinUCB():
    
    def __init__(self, dataset, alpha):
        self.data = dataset
        self.d = len(dataset[0][0])
        
        self.alpha = alpha
        
        self.actions = ['low', 'mid', 'high']
        
        self.A = {'low': np.identity(self.d),
                  'mid': np.identity(self.d),
                  'high': np.identity(self.d)}
        self.b = {'low': np.zeros(self.d),
                  'mid': np.zeros(self.d),
                  'high': np.zeros(self.d)}
        self.beta = {'low': 0,
                     'mid': 0,
                     'high': 0}
        
        self.t = 0
        self.total_regret = 0
        self.regret = []
        
    def ucb(self, features, a):
        return np.dot(features, self.beta[a]) + self.alpha * np.sqrt(np.dot(features,
                     np.matmul(np.linalg.inv(self.A[a]),
                               np.transpose(features))))
    
    def argmax(self, features):
        best_a = None
        best_v = - float('inf')
        
        for a in self.actions:
            ucb = self.ucb(features, a)
            
            if ucb == best_v and random.random() < 0.5:
                best_a = a
                best_v = ucb
            
            if ucb > best_v:
                best_a = a
                best_v = ucb
            
        return best_a
    
    def reward(self, action, dose):
        if action == dose:
            return 0
        
        else:
            return -1
    
    def run(self):
        
        for features, dose in self.data:
            print(self.t)
            
            for a in self.actions:
                self.beta[a] = np.matmul(np.linalg.inv(self.A[a]), np.transpose(self.b[a]))
            
            action = self.argmax(features)
            print(action)
            
            reward = self.reward(action, dose)
            print(reward)
            
            self.A[action] = self.A[action] + np.outer(features, features)
            self.b[action] = self.b[action] + reward * features
            
            self.t += 1
            self.total_regret -= reward
            self.regret.append(self.total_regret)