# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:39:07 2019

@author: Lucas
"""

def convert(dose):
    dose = float(dose)
    
    if dose/7 < 3:
        return 'low'
    
    elif dose/7 <= 7:
        return 'mid'
    
    else:
        return 'high'

def decade(age):
    return float(age[0])