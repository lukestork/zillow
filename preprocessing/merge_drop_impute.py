#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 21:40:05 2017

@author: Luke
"""

import pandas as pd
import numpy as np
import scipy as sp
import math
import statistics as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

write=1
#Load properties data from .csv file in workspace
prop_data = pd.read_csv('/python_files_luke/zillow_datasets/properties_2016.csv')
#Load train log_error and dates data
train_data = pd.read_csv('/python_files_luke/zillow_datasets/train_2016_v2.csv')


##
#Gets into feature selection, dropping columns etc.
##

#Features are frequently nan, sometimes entire columns calculate percent
# missing of each feature
num_cols=len(prop_data.columns)
missing_data = pd.DataFrame(prop_data.isnull().sum() / float(len(prop_data)),columns=['nan_pct'])
missing_data.sort_values(by='nan_pct',ascending=False).head(num_cols)

#Get the indices of features that have above %70 missing and remove them 
bad_feature_indices=np.where(missing_data.nan_pct > .5)
prop_data.drop(prop_data.columns[list(bad_feature_indices)],axis=1,inplace=True)


#For the first iteration, impute any missing value with the mode
imputed_data=prop_data.fillna(prop_data.median())


if write==1:
    #Write test data and labels to csv
    imputed_data.to_csv('test_data.csv')

#merge dataframes to create large set
merged_data=pd.merge(prop_data, train_data, on='parcelid', how='left')
#drop duplicate parcelids that have multiple or erroneous transactions
#this is to be done as stated from Kaggle moderators
all_data_no_repeats=merged_data.drop_duplicates(subset='parcelid', 
                                                keep='first', 
                                                inplace=False)

#Once data is merged, pull out all of the data that has training labels
#<100 is just a stupid filter criteria
merge_subset_train_data=merged_data.loc[merged_data['logerror'] < 100]
all_labels=merge_subset_train_data.logerror
#Drop labels from associated data and generate test train sets
del merge_subset_train_data['logerror']
imputed_train_data=merge_subset_train_data.fillna(prop_data.median())


if write==1:
    #Write train data and labels to csv
    imputed_train_data.to_csv('merged_train_data.csv')
    all_labels.to_csv('train_labels.csv')

all_labels.to_csv('train_labels.csv', header='True')

