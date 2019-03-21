# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:12:28 2019

@author: Lucas
"""

import baseline, linucb, lasso
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# sample data #

dataset = linucb.dataset_3 # changes according to the algorithm considered

mean = []
z = 1.96 # 95% confidence
low_ci = []
upper_ci = []
base_mean = []

N = 1
data_points = []
base_data_points = []

for _ in range(N):
    random.shuffle(dataset)
    alg = linucb.LinUCB(dataset=dataset, alpha=0.01) # changes according to the algorithm considered
    alg.run()
    data_points.append(alg.regret)
    
    base_data_points_k = []
    reg = 0
    for feature, dose in dataset:
        if dose != 'mid':
            reg += 1
        base_data_points_k.append(reg)
    base_data_points.append(base_data_points_k)

# create performance quantities #

for t in range(len(dataset)):
    v = []
    
    for k in range(N):
        v.append(data_points[k][t])
    est = np.mean(v)
    mean.append(est)
    
    s = np.std(v)
    low_ci.append(est - z * s / np.sqrt(N))
    upper_ci.append(est + z * s / np.sqrt(N))

for t in range(len(dataset)):
    v = []
    
    for k in range(N):
        v.append(base_data_points[k][t])
    est = np.mean(v)
    base_mean.append(est)
    print('lol')
    
# generate plot of performance #
        
def lineplotCI(x_data, y_data, sorted_x, low_CI, upper_CI, x_label, y_label, title, vs):
    _, ax = plt.subplots()
    ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'LinUCB')
    ax.fill_between(sorted_x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
    ax.plot(x_data, vs, lw = 1, color = 'red', alpha = 1, label = 'Baseline (Fixed Dosing)')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc = 'best')

print(mean)
print(base_mean)

lineplotCI(x_data = [t for t in range(len(dataset))]
           , y_data = mean
           , sorted_x = [t for t in range(len(dataset))]
           , low_CI = low_ci
           , upper_CI = upper_ci
           , x_label = 'Number of patients seen'
           , y_label = 'Regret'
           , title = 'Regret vs. Number of patients seen (online)'
           , vs = base_mean)