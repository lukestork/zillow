#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 21:35:12 2017

@author: Luke
"""

import pandas as pd
import numpy as np
import scipy as sp
import math
import statistics as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



#load .csv files that were generated from write=1 in merge_drop_impute.py
#Load preprocessed train data from .csv file in workspace
merged_train_data = pd.read_csv('/python_files_luke/zillow_datasets/merged_train_data.csv')
del merged_train_data['transactiondate']
del merged_train_data['propertycountylandusecode']
del merged_train_data['propertyzoningdesc']
del merged_train_data['Unnamed: 0']

#load training labels
train_labels = pd.read_csv('/python_files_luke/zillow_datasets/train_labels.csv')

#Load large submission test data to predict
submission_test_data= pd.read_csv('/python_files_luke/zillow_datasets/test_data.csv')
del submission_test_data['propertycountylandusecode']
del submission_test_data['propertyzoningdesc']
del submission_test_data['Unnamed: 0']

#Load sample submission for formatting
sample_submission = pd.read_csv('/python_files_luke/zillow_datasets/sample_submission.csv')

#Crude way of writing to .csv leaves unwanted column of old indices, drop them
del train_labels['Unnamed: 0']

#split into test/train sets
X_train, X_test, y_train, y_test = train_test_split(merged_train_data, 
                                                    train_labels, 
                                                    test_size=0.2, 
                                                    random_state=42)

#Regression random forest classifier
regr = RandomForestRegressor(max_depth=7, random_state=0)
regr.fit(X_train, y_train.values.ravel())
test_output=regr.predict(X_test)

error=mean_absolute_error(y_test.values,test_output)

submission_output=regr.predict(submission_test_data)

#TODO
#write submission output to sample_submission dataframe and save as .csv
