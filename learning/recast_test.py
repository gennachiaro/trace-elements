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
df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64}, na_values= na_values, index_col=1)
df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64}, na_values= na_values)

# drop blank columns
df = df.dropna(axis = 1, how = 'all')
df1 = df1.dropna(axis = 1, how = 'all')

# drop rows with any NaN values
#df = df.dropna()
#df1 = df1.dropna()

# fill NaN
#df = df.fillna(method = 'bfill')
#df1 = df1.fillna(method = 'bfill')

#change all negatives to zeroes
num = df._get_numeric_data()
num[num < 0] = 0

num = df1._get_numeric_data()
num[num < 0] = 0

#map out columns
col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df.columns)]

#DataFrameMelt
melt = (df.melt(id_vars=['Spot', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))
print(melt)

#mean = df.mean(level = 'Sample')

mean = df1.groupby('Sample').mean()
print(mean.head())

#Create dataframe with sample populations
populations = df1[['Sample','Population']].drop_duplicates('Sample')

# Merge two dataframes
merge = pd.merge(populations, mean, how = 'right', left_on= "Sample", right_on = mean.index)

print (merge.head())


#DataFrameMelt for tidy data and plotting final values

melt2 = (merge.melt(id_vars=['Sample', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr','SiO2'], ignore_index=False))

#melt2 = melt2.set_index('Sample')

melt2.groupby(['Sample']).mean()

print(melt2)
