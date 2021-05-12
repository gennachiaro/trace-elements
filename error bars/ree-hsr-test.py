#~/opt/anaconda3/bin/myenv
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
# read in excel file!

# All with clear mineral analyses removed
df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.xlsx',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values, sheet_name = 'Data')

# Drop "Included" column
df1 = df1.drop(['Included'], axis = 1)

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
melt = (df1.melt(id_vars=['Sample','Spot', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))
#print(melt)

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

# Set index
merge = merge.set_index('Sample')
#print (merge.head())

# Calculate stdev for each sample (messy)
sample_std = df1.groupby('Sample').std()

# Multiply dataframe by two to get 2 sigma
#sample_std = sample_std *2
#print(sample_std.head())

# Merge two dataframes (stdev and populations)
sample_std = pd.merge(populations, sample_std, how = 'right', left_on= "Sample", right_on = sample_std.index)
#print (sample_std.head())

# Set index
sample_std = sample_std.set_index('Sample')
#print (sample_std.head())

# Plotting
#       Slicing dataframe with .loc
#MG = merge.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#MG_index = MG.index

#VCCR = merge.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#VCCR_index = VCCR.index

# Dataframe Slicing using "isin"
VCCR = melt[melt['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG = melt[melt['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]

VCCRREE = VCCR[VCCR['variable'].isin(['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Yb', 'Lu'])]
MGREE = MG[MG['variable'].isin(['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Yb', 'Lu'])]

# Drop bad analyses column
#MG = MG.set_index('Sample')
#MG = MG.drop(['ORA-2A-004','ORA-2A-004-B', 'ORA-2A-036', 'ORA-2A-036-B'], axis = 0)

#MG = MG.reset_index()

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,3.5))

#group plot title
title = fig.suptitle("High-Silica Rhyolite (VCCR + MG) Fiamme Glass", fontsize=16, y = 1.03)

#plot 1 

#create ree plot
plt.subplot(1,2,1)
plot = sns.lineplot(data = MGREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief")
plot1 = sns.lineplot(data = VCCRREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
#plot just populations
plt.legend(h[1:4]+h[5:9],l[1:4]+l[5:9],loc='lower right')

plt.ylabel=("Sample/Chondrite")
plt.ylim(0.05, 200)

#set y axis to log scale
plot1.set(yscale='log')

fig.tight_layout()
plt.show()

