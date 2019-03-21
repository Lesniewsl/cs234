# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:28:04 2019

@author: Lucas
"""

import numpy as np
from sklearn import linear_model
from numpy import random
from lasso_data import dataset_4
import matplotlib.pyplot as plt

# lasso bandit #

class Lasso():
    
    def __init__(self, dataset, q, h, lam1, lam20):
        self.data = dataset
        self.d = len(dataset[0][0])
        
        self.q = q # forced samplinig parameter
        self.h = h # localization parameter
        self.lam1 = lam1
        self.lam20 = lam20
        self.lam2 = lam20
        
        self.actions = ['low', 'mid', 'high']
        
        self.T = {'low': [],
                  'mid': [],
                  'high': []}
        self.X_forced = {'low': [],
                  'mid': [],
                  'high': []}
        self.Y_forced = {'low': [],
                  'mid': [],
                  'high': []}
        
        self.S = {'low': [],
                  'mid': [],
                  'high': []}
        self.X_all = {'low': [],
                  'mid': [],
                  'high': []}
        self.Y_all = {'low': [],
                  'mid': [],
                  'high': []}
        
        self.t = 0
        self.total_regret = 0
        self.regret = []
        
    def forced_sample_times(self):
        mapping = {'low': 1, 'mid': 2, 'high': 3}
        
        for n in range(np.sqrt(len(self.data) + 1)):
            for a in self.actions:
                for j in range(self.q * (mapping[a] - 1) + 1, self.q * mapping[a] + 1):
                    self.T[a].append((2**n - 1) * 3 * self.q + j)
    
    def lasso_beta_forced(self, a):
        if self.X_forced[a] == []:
            return np.zeros(self.d)
        
        else:
            lasso = linear_model.Lasso(alpha=self.lam1, fit_intercept=False)
            lasso.fit(self.X_forced[a], self.Y_forced[a])
            return lasso.coef_
    
    def lasso_beta_all(self, a):
        if self.X_all[a] == []:
            return np.zeros(self.d)
        
        else:
            lasso = linear_model.Lasso(alpha=self.lam2, fit_intercept=False)
            lasso.fit(self.X_all[a], self.Y_all[a])
            return lasso.coef_
    
    def argmax(self, features, action_set):
        best_a = None
        best_v = - float('inf')
        
        for a in action_set:
            val = np.dot(features, self.lasso_beta_all(a))
            
            if val == best_v and random.random() < 0.5:
                best_a = a
                best_v = val
            
            if val > best_v:
                best_a = a
                best_v = val
            
        return best_a
    
    def reward(self, action, dose):
        if action == dose:
            return 0
        
        else:
            return -1
    
    def run(self):
        for features, dose in self.data:
            print(self.t)
            
            action = None
            
            for a in self.actions:
                if self.t in self.T[a]:
                    action = a
                    reward = self.reward(action, dose)
                     
                    self.X_forced[action].append(features)
                    self.Y_forced[action].append(reward)
            
            if action == None:
                 beta = {'low': 0, 'mid': 0, 'high': 0}
                 best_v = - float('inf')
                 for a in self.actions:
                     beta[a] = np.dot(features, self.lasso_beta_forced(a))
                     if beta[a] > best_v:
                         best_v = beta[a]
                 
                 action_set = []
                 for a in self.actions:
                     if beta[a] >= best_v - self.h/2:
                         action_set.append(a)
                         
                 action = self.argmax(features, action_set)
            
            print(action)
            reward = self.reward(action, dose)
            print(reward)
            
            self.S[action].append(self.t)
            self.X_all[action].append(features)
            self.Y_all[action].append(reward)
            self.lam2 = self.lam20 * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
            
            self.t += 1
            self.total_regret -= reward
            self.regret.append(self.total_regret)

if __name__ == '__main__':
    random.shuffle(dataset_4)
    alg = Lasso(dataset=dataset_4, q=1, h=5, lam1=0.05, lam20=0.05)
    alg.run()
    plt.plot(alg.regret)
    