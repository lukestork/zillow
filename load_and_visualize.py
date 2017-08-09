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

#Load properties data from .csv file in workspace
properties_data = pd.read_csv('/Users/Luke/python_files_luke/zillow_datasets/properties_2016.csv')

#p = ggplot(aes(x='something', y='something_else'), data=properties_data.something)