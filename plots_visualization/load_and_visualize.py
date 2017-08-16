#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug  8 
@author: Luke
This script doesn't do too much at this point. Makes a couple plots, but the most 
useful plots were found on the kaggle discussions. This will most likely be
more useful when engineering features
"""
import pandas as pd
import numpy as np
import scipy as sp
import math
import statistics as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ggplot import *
#ggplot dumps a bunch of the stupidest data upon import, delete it for now
del diamonds, chopsticks, meat, movies, mpg, mtcars, pageviews, pigeons, salmon



#Load properties data from .csv file in workspace
prop_data = pd.read_csv('/python_files_luke/zillow_datasets/properties_2016.csv')
#Load train log_error and dates data
train_data = pd.read_csv('/python_files_luke/zillow_datasets/train_2016_v2.csv')


#Simple plots for features just to get a feel
p = ggplot(aes( x='calculatedfinishedsquarefeet',y='bathroomcnt'), data=prop_data)
p + geom_point() 
print(p)

p = ggplot(aes( x='yearbuilt',y='lotsizesquarefeet'), data=prop_data)
p + geom_point() + scale_y_continuous(limits=(0,10**7))
print(p)


