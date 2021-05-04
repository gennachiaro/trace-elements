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
df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64}, na_values= na_values)

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
#col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df1.columns)]

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
all = (df1.melt(id_vars=['Sample','Spot', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))
print(all)

# Indexing
all = all.set_index('Sample')


# Calculate means for each sample (messy)
#sample_mean = df1.groupby('Sample').mean()
#print(sample_mean.head())

# Another way to calculate means, but need an indexed df to use "level"
#   we need to use a non-indexed dataframe to make our "populations" dataframe 
#       so might as well just use one dataframe and the groupby fx to make it simpler
#mean = df.mean(level = 'Sample') # another way to calculate means, but need an indexed df to use "level"

# Create seperate dataframe with sample populations
#   this is so we can merge this with the mean dataframe because the groupby fx just gives sample name and value for each element
#populations = df1[['Sample','Population']].drop_duplicates('Sample')

# Merge two dataframes
#merge = pd.merge(populations, sample_mean, how = 'right', left_on= "Sample", right_on = sample_mean.index)
#print (merge.head())

# DataFrameMelt for tidy data and plotting final values
#means = (merge.melt(id_vars=['Sample', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))

# Indexing
#means = means.set_index('Sample')

# Multi-Indexing
#means = means.set_index(['Sample', 'variable']).sort_index()

#print(means.head())

# Plotting
#       Slicing dataframe
#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
#FGCP_index = FGCP.index

MG = all.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

VCCR = all.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
VCCR_index = VCCR.index

#VCCR1 = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#VCCR1_index = VCCR1.index

#MG1 = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#MG1_index = MG1.index

#FG = df.loc [['ORA-5B-410','ORA-5B-412A-FG','ORA-5B-412B-FG','ORA-5B-414-FG']]
#FG_index = FG.index

# Set background color
sns.set_style("darkgrid")

# Set axes limits
#plt.ylim(10, 50)
#plt.xlim (-0.2,0.4)

# Set color palette
#sns.set_palette("PuBuGn_d")

# Create plot
#plot = sns.scatterplot(data = VCCR1, x= 'Ba', y='Sr',hue = "Population",  style= VCCR1_index, palette="binary", marker = 'x', edgecolor="black", s=150, legend = False, alpha = 0.4, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
#plot = sns.scatterplot(data = MG1, x= 'Ba', y='Sr',hue = "Population" , style = "Population", palette="binary",marker = 'x', edgecolor="black", s=150,alpha = 0.3, legend = "brief")

#plot = sns.scatterplot(data = MG, x= MG[MG['variable'] == 'Sr'], y= MG[MG['variable'] == 'Y'],hue = "Population" , style = MG, palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.5, legend = "brief")
#plot = sns.scatterplot(data = VCCR, x= 'Sr', y='Y',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], legend = "brief")




#plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)

# Set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

# Set location of legend
#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# Configure legend
h,l = plot.get_legend_handles_labels()
# Populations
plt.legend(h[1:4]+h[13:16],l[1:4]+l[13:16],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# samples + populations
#plt.legend(h[0:4]+ h[5:12]+h[13:16]+h[17:27],l[0:4]+l[5:12]+ l[13:16]+l[17:27],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# General title
plt.suptitle("High-Silica Rhyolite (MG + VCCR) Fiamme Glass", fontsize=15, fontweight=0, color='black')

# Set size of plot
sns.set_context("paper")

plt.show()