#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug  8 
@author: Luke
"""
import pandas as pd
import numpy as np
import scipy as sp
import math
import statistics as stats
import matplotlib.pyplot as plt

from ggplot import *
#ggplot dumps a bunch of stupidest data upon import, delete it for now
del diamonds, chopsticks, meat, movies, mpg, mtcars, pageviews, pigeons, salmon

#Load properties data from .csv file in workspace
properties_data = pd.read_csv('/python_files_luke/zillow_datasets/properties_2016.csv')

#Many thousand rows contain all nan with just parcelid
#Just get the indices for these rows for now
num_cols=len(properties_data.columns)
nan_list=properties_data.isnull().sum(axis=1).tolist()
nan_array= np.asarray(nan_list)
all_nan_indices=np.where(nan_array == num_cols-1)

#TODO:
#Remove nans from properties_data using all_nan_indices

#Simple plots for features just to get a feel
p = ggplot(aes( x='calculatedfinishedsquarefeet',y='bathroomcnt'), data=properties_data)
p + geom_point() + scale_x_continuous(limits=(40000))
print(p)

p = ggplot(aes( x='yearbuilt',y='lotsizesquarefeet'), data=properties_data)
p + geom_point() + scale_y_continuous(limits=(0,10**7))
print(p)


