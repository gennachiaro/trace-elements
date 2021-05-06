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
    #All values
#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64}, na_values= na_values)

# Create a custom list of values I want to cast to NaN, and explicitly 
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values)

# All with clear mineral analyses removed (excel file!)
#df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.xlsx',skiprows = [2],dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values, sheet_name = 'Data')

#df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx',skiprows = [2],dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values, sheet_name = 'Data')

# Columns converted to np.float
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# All with clear mineral analyses removed (excel file!)
#df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx',sheet_name = 'Data', skiprows = [1], na_values = na_values)
df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx',sheet_name = 'Data', skiprows = [1], dtype = data, na_values = na_values)

#df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx',sheet_name = 'Data', skiprows = [1], dtype = dtype, na_values = na_values)

# Element columns converted to 'float.64'
#dtype = {'Li': 'float64', 'Mg': 'float64', 'Al': 'float64','Si': 'float64','Ca': 'float64','Sc': 'float64','Ti': 'float64','Ti.1': 'float64','V': 'float64', 'Cr': 'float64', 'Mn': 'float64','Fe': 'float64','Co': 'float64','Ni': 'float64', 'Zn': 'float64','Rb': 'float64','Sr': 'float64','Y': 'float64','Zr': 'float64','Nb': 'float64','Ba': 'float64','La': 'float64','Ce': 'float64','Pr': 'float64','Nd': 'float64','Sm': 'float64','Eu': 'float64','Gd': 'float64','Tb': 'float64','Gd.1': 'float64','Dy': 'float64','Ho': 'float64','Er': 'float64','Tm': 'float64','Yb': 'float64','Lu': 'float64','Hf': 'float64','Ta': 'float64','Pb': 'float64','Th': 'float64','U': 'float64'}

# Element columns converted to np.float
#dtype = {'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64}

# Element columns changed with .astype to 'float.64'
#df1.astype({'Li': 'float64', 'Mg': 'float64', 'Al': 'float64','Si': 'float64','Ca': 'float64','Sc': 'float64','Ti': 'float64','Ti.1': 'float64','V': 'float64', 'Cr': 'float64', 'Mn': 'float64','Fe': 'float64','Co': 'float64','Ni': 'float64', 'Zn': 'float64','Rb': 'float64','Sr': 'float64','Y': 'float64','Zr': 'float64','Nb': 'float64','Ba': 'float64','La': 'float64','Ce': 'float64','Pr': 'float64','Nd': 'float64','Sm': 'float64','Eu': 'float64','Gd': 'float64','Tb': 'float64','Gd.1': 'float64','Dy': 'float64','Ho': 'float64','Er': 'float64','Tm': 'float64','Yb': 'float64','Lu': 'float64','Hf': 'float64','Ta': 'float64','Pb': 'float64','Th': 'float64','U': 'float64'}).dtypes

# Drop blank columns
#df = df.dropna(axis = 1, how = 'all')
df1 = df1.dropna(axis = 1, how = 'all')

# Drop Blank Rows
#   Drops fully blank rows
df1 = df1.dropna(axis = 0, how = 'all')

# Need to drop rows that aren't useful but don't drop rows with nAn values for only some elements

# Drop Stats Rows
#df1 = df1.drop(['Median, 'IQR','Mean','St Dev','Rel Error','Mean (Outliers excluded)', 'StDev (Outliers excluded)', 'Rel Error (Outliers excluded)'], axis = 0)

# Set index for dropping valies
df1 = df1.set_index('Spot')
#df1 = df1.drop(['Median'], axis = 0)
df1 = df1.drop(['Median', 'IQR','Mean','St Dev','Rel Error','Mean (Outliers excluded)', 'StDev (Outliers excluded)', 'Rel Error (Outliers excluded)', 'ORA-5B-405-B', 'ORA-5B-406-B', 'ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-2A-004-B', 'ORA-2A-036-B', 'GlassSlide', 'Epoxy', 'ORA-2A-002-Type3', 'ORA-2A-002-Type2'], axis = 0)

# Drop fully blank rows
df1 = df1.dropna(axis = 0, how = 'all')
df1 = df1.reset_index()

# Drop secondary index
df1 = df1.set_index('Sample')
df1 = df1.drop(['B = bad first run'], axis = 0)

# Reset Index
df1 = df1.reset_index()

# Drop "Included" column
df1 = df1.drop(['Included'], axis = 1)

# Drop Row 2 
#df1 = df1.drop([2], axis = 1)

# Drop Specific Columns
#df1 = df1.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B','ORA-5B-416-B'], axis = 1)

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
#col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df1.columns)]

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
all = (df1.melt(id_vars=['Sample','Spot', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))
#print(all)

# Calculate means for each sample (messy)
sample_mean = df1.groupby('Sample').mean()
#print(sample_mean.head())

# Another way to calculate means, but need an indexed df to use "level"
#   we need to use a non-indexed dataframe to make our "populations" dataframe 
#       so might as well just use one dataframe and the groupby fx to make it simpler
#mean = df.mean(level = 'Sample') # another way to calculate means, but need an indexed df to use "level"

# Create seperate dataframe with sample populations
#   this is so we can merge this with the mean dataframe because the groupby fx just gives sample name and value for each element
populations = df1[['Sample','Population']].drop_duplicates('Sample')

# Merge two dataframes
merge = pd.merge(populations, sample_mean, how = 'right', left_on= "Sample", right_on = sample_mean.index)
print (merge.head())

# DataFrameMelt for tidy data and plotting final values
means = (merge.melt(id_vars=['Sample', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))

# Indexing
#means = means.set_index('Sample')

# Multi-Indexing
#means = means.set_index(['Sample', 'variable']).sort_index()

print(means.head())