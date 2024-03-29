#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:32:52 2019

@author: gennachiaro
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from PIL.Image import merge

# import csv file
# all values
#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv')

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
#df1 = df1.dropna(axis = 1, how = 'all')

#indexed dataframe for 
#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', index_col=1)

#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv')

# note: the dtypes dictionary specifying types. pandas will attempt to infer
#   the type of any column name that's not listed

#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64})


# note the new na_values argument provided to read_csv
#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', dtype={'Li': np.float64}, na_values= ['<-1.00'])


# Create a custom list of values I want to cast to NaN, and explicitly 
#   define the data types of columns:

na_values = ['<-1.00', '****', '<****']
df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64}, na_values= na_values)

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
#df1 = df1.dropna(axis = 1, how = 'all')

# drop rows with any NaN values
#df = df.dropna()
#df1 = df1.dropna()

# fill NaN with zeroes
#df = df.fillna(method = 'bfill')
#df1 = df1.fillna(method = 'bfill')

#df.iloc[3:62][df.iloc[3:62]< 0] = 0

#df = np.where(df<0, 0, df)

#num = df._get_numeric_data()
#num[num < 0] = 0

#num = df1._get_numeric_data()
#num[num < 0] = 0

#map out columns
col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df.columns)]

