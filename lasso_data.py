# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:27:19 2019

@author: Lucas
"""

# dataset 4 (lasso bandit) #

import numpy as np
import pandas as pd
from sklearn import preprocessing
import csv
from utils import convert

df = pd.read_csv('warfarin.csv')

df = df[['Gender',
         'Race',
         'Ethnicity',
         'Age',
         'Indication for Warfarin Treatment',
         'Diabetes',
         'Congestive Heart Failure and/or Cardiomyopathy',
         'Valve Replacement',
         'Aspirin',
         'Acetaminophen or Paracetamol (Tylenol)',
         'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day',
         'Simvastatin (Zocor)',
         'Atorvastatin (Lipitor)',
         'Fluvastatin (Lescol)',
         'Lovastatin (Mevacor)',
         'Pravastatin (Pravachol)',
         'Rosuvastatin (Crestor)',
         'Cerivastatin (Baycol)',
         'Amiodarone (Cordarone)',
         'Carbamazepine (Tegretol)',
         'Phenytoin (Dilantin)',
         'Rifampin or Rifampicin',
         'Cyp2C9 genotypes',
         'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
         'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
         'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
         'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
         'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
         'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
         'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C']].astype(str)

le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)

enc = preprocessing.OneHotEncoder()
enc.fit(df)
onehotlabels = enc.transform(df).toarray()

df2 = pd.DataFrame(onehotlabels)

with open('warfarin.csv') as f:
    next(f)
    read_csv = csv.reader(f, delimiter=',')
    dataset_4 = []
    
    for row in read_csv:
        dose = row[34]
        
        if dose != 'NA' and dose != '':
            dose = convert(dose)
            
            data = np.array(df2.loc[[read_csv.line_num - 1]])
            features = np.append(1, data) # bias term
            
            dataset_4.append((features, dose))
