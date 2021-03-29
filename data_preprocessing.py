# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:23:22 2021

@author: NathanWaddell
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)