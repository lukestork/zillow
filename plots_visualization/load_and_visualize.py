#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug  8 
@author: Luke
"""
import pandas as pd
import numpy as np
import random 
random.seed(42)
import scipy as sp
import math
import statistics as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ggplot import *
#ggplot dumps a bunch of stupidest data upon import, delete it for now
del diamonds, chopsticks, meat, movies, mpg, mtcars, pageviews, pigeons, salmon

#Load properties data from .csv file in workspace
prop_data = pd.read_csv('/python_files_luke/zillow_datasets/properties_2016.csv')
#Load train log_error and dates data
train_data = pd.read_csv('/python_files_luke/zillow_datasets/train_2016_v2.csv')

#merge dataframes to create large set
merged_data=pd.merge(prop_data, train_data, on='parcelid', how='left')
#drop duplicate parcelids that have multiple or erroneous transactions
#this is to be done as stated from Kaggle
all_data_no_repeats=merged_data.drop_duplicates(subset='parcelid', 
                                                keep='first', 
                                                inplace=False)

#Once data is merged, pull out all of the data that has training labels
#<100 is just a stupid filter criteria
merge_subset_train_data=merged_data.loc[merged_data['logerror'] < 100]
all_labels=merge_subset_train_data.logerror
#Drop labels from associated data and generate test train sets
del merge_subset_train_data['logerror']
X_train, X_test, y_train, y_test = train_test_split(merge_subset_train_data, 
                                                    all_labels, 
                                                    test_size=0.2, 
                                                    random_state=42)

##
#Gets into feature selection, dropping columns etc.
#The below section should probably end up before the test train split, can be modified later
##

#Features are frequently nan, sometimes entire columns calculate percent
# missing of each feature
num_cols=len(prop_data.columns)
missing_data = pd.DataFrame(prop_data.isnull().sum() / float(len(prop_data)),columns=['nan_pct'])
missing_data.sort_values(by='nan_pct',ascending=False).head(num_cols)

#Get the indices of features that have above %60 missing and remove them 
bad_feature_indices=np.where(missing_data.nan_pct > .6)
prop_data.drop(prop_data.columns[list(bad_feature_indices)],axis=1,inplace=True)

#Many thousand rows contain all nan with just parcelid
#Just get the indices for these rows for now
num_cols=len(prop_data.columns)
nan_list=prop_data.isnull().sum(axis=1).tolist()
nan_array= np.asarray(nan_list)
all_nan_indices=np.where(nan_array >= num_cols-1)

#Remove nans from prop_data using all_nan_indices
reduced_properties=prop_data.drop(prop_data.index[list(all_nan_indices)])

#Simple plots for features just to get a feel
p = ggplot(aes( x='calculatedfinishedsquarefeet',y='bathroomcnt'), data=prop_data)
p + geom_point() 
print(p)

p = ggplot(aes( x='yearbuilt',y='lotsizesquarefeet'), data=prop_data)
p + geom_point() + scale_y_continuous(limits=(0,10**7))
print(p)


