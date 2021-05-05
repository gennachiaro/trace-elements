
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

# Import csv file

# Create a custom list of values I want to cast to NaN, and explicitly 
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values)

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
df1 = df1.dropna(axis = 1, how = 'all')

# NaN treatment:
#   change all negatives and zeroes to NaN
num = df1._get_numeric_data()
num[num <= 0] = np.nan

#   change all negatives to NaN
#num = df1._get_numeric_data()
#num[num < 0] = np.nan

#   drop rows with any NaN values
#df = df.dropna()
#df1 = df1.dropna()

#   fill NaN
#df = df.fillna(method = 'bfill')
#df1 = df1.fillna(method = 'bfill')

# map out columns
col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df1.columns)]

# Need to get dataframes of each column, groupby sample name, and calculate values

# put variable for column name and find a way to iterate through the columns

#for c in enumerate(df1.columns)
x = df1[['Sample','Li']]

df1 = df1.set_index('Sample')

#print(df1.head())
# Series output
#x = df1.loc['ORA-2A-001','Li']

# Dataframe Output
x = df1.loc[['ORA-2A-001','ORA-2A-035'],['Li','Mg']]
print(x)

sample_mean = x.groupby('Sample').mean()

#print(sample_mean.head())

iqr = x.groupby('Sample').quantile()

#print(iqr.head())


# find a way to do dataframe algebra so it is if > median + Iqr

#print(df1.info())