# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:41:22 2019

@author: Lucas
"""

import numpy as np
from numpy import random
from baseline_data import dataset_1, dataset_2
from utils import convert

# fixed dose #

count = 0
N = len(dataset_1)

for dose in dataset_1:
    if dose == 'mid':
        count += 1/N

print('perf fixed dose on dataset #1: ' + str(count))
      
count = 0
N = len(dataset_2)

for features, dose in dataset_2:
    if dose == 'mid':
        count += 1/N

print('perf fixed dose on dataset #2: ' + str(count))

# clinical dosing algorithm #

weights = np.array([4.0376,
           - 0.2546,
           0.0118,
           0.0134,
           - 0.6752,
           0.4060,
           0.0443,
           1.2799,
           - 0.5965])

count = 0
N = len(dataset_2)

for features, dose in dataset_2:
    pred = convert(np.dot(features, weights)**2)
    if dose == pred:
        count += 1/N

print('perf clinical dosing algorithm on dataset #2: ' + str(count))