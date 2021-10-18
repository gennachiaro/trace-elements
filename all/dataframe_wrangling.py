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

# Create a custom list of values I want to cast to NaN, and explicitly 
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

# Columns converted to np.float
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx'
path = os.path.normcase(path) # This changes path to be compatible with windows

# Master spreadsheet with clear mineral analyses removed (excel file!)
df1 = pd.read_excel(path, sheet_name = 'Data', skiprows = [1], dtype = data, na_values = na_values)

#-----------------------------
# Cleaning up the spreadsheet:

# Drop fully blank columns
df1 = df1.dropna(axis = 1, how = 'all')

# Drops fully blank rows (spacing rows)
#   and rows that just have sample name
df1 = df1.set_index('Spot')
df1 = df1.dropna(axis = 0, how = 'all')

# Drop Stats Rows
df1 = df1.drop(['Median', 'IQR','Mean','St Dev','Rel Error','Mean (Outliers excluded)', 'StDev (Outliers excluded)', 'Rel Error (Outliers excluded)'], axis = 0)

# Drop Samples:
# Analyses we measured twice because laser was actin' up
df1 = df1.drop(['ORA-5B-405-B', 'ORA-5B-406-B', 'ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-2A-004-B', 'ORA-2A-036-B'], axis = 0)

# Don't remember why we delete these but we are
df1 = df1.drop(['ORA-2A-002-Type3', 'ORA-2A-002-Type2'], axis = 0)

# Measured slide and epoxy values for comparison
df1 = df1.drop(['GlassSlide', 'Epoxy'], axis = 0)
df1 = df1.reset_index() 

#If Value in included row is equal to zero, drop this row
df1 = df1.loc[~((df1['Included'] == 0))]

# Drop "Included" column
df1 = df1.drop(['Included'], axis = 1)

#-----------------------------
# NaN treatment:

# Change only negatives to NaN
num = df1._get_numeric_data()
num[num < 0] = np.nan

# Change all negatives and zeroes to NaN
# num = df1._get_numeric_data()
# num[num <= 0] = np.nan

#   drop rows with any NaN values
#df = df.dropna()
#df1 = df1.dropna()

#   fill NaN
#df = df.fillna(method = 'bfill')
#df1 = df1.fillna(method = 'bfill')

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