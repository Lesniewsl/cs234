# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:22:41 2019

@author: Lucas
"""

import numpy as np
import csv
from utils import convert, decade

# dataset 1 (fixed dose) #

with open('warfarin.csv') as f:
    next(f)
    read_csv = csv.reader(f, delimiter=',')
    dataset_1 = []
    
    for row in read_csv:
        dose = row[34]
        
        if dose != 'NA' and dose != '':
            dose = convert(dose)
            dataset_1.append(dose)

# dataset 2 (fixed dose + clinical dosing algorithm) #

with open('warfarin.csv') as f:
    next(f)
    read_csv = csv.reader(f, delimiter=',')
    dataset_2 = []
    
    for row in read_csv:
        dose = row[34]
        
        if dose != 'NA' and dose != '':
            dose = convert(dose)
            
            if (row[4] != 'NA' and row[4] != '' and
                row[5] != 'NA' and row[5] != '' and
                row[6] != 'NA' and row[6] != ''):
                
                features = []
                features.append(1) # bias term
                
                age = decade(row[4])
                features.append(age)
                
                height = float(row[5])
                features.append(height)
                
                weight = float(row[6])
                features.append(weight)
                
                asian = int(row[2] == 'Asian')
                features.append(asian)
                
                black = int(row[2] == 'Black or African American')
                features.append(black)
                
                missing = int(row[2] == 'Unknown' or
                              row[2] == 'NA')
                features.append(missing)
                
                enzyme = int(row[24] == 1 or
                             row[25] == 1 or
                             row[26] == 1)
                features.append(enzyme)
                
                amio = int(row[23] == 1)
                features.append(amio)
                
                features = np.array(features)
                dataset_2.append((features, dose))