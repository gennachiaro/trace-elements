#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:32:5 2 2019

@author: gennachiaro
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values)
# read in excel file!

# # All data with clear mineral analyses removed (manually in excel file)
# df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.xlsx', dtype=({
#                     'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64, 'SiO2': np.float64}), na_values=na_values, sheet_name='Data')


# Columns converted to np.float
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx'

#path = os.path.normcase(path) # This changes path to be compatible with windows

# Master spreadsheet with clear mineral analyses removed (excel file!)
df1 = pd.read_excel(path, sheet_name = 'Data', skiprows = [1], dtype = data, na_values = na_values)

#If Value in included row is equal to zero, drop this row
df1 = df1.loc[~((df1['Included'] == 0))]

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
df1 = df1.dropna(axis=1, how='all')

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
melt = (df1.melt(id_vars=['Sample', 'Spot', 'Population'], value_vars=['Li', 'Mg', 'Al', 'Si', 'Ca', 'Sc', 'Ti', 'Ti.1', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
        'Tb', 'Gd.1', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U', 'Rb/Sr', 'Ba/Y', 'Zr/Y', 'Zr/Ce', 'Zr/Nb', 'U/Ce', 'Ce/Th', 'Rb/Th', 'Th/Nb', 'U/Y', 'Sr/Nb', 'Gd/Yb', 'U/Yb', 'Zr/Hf', 'Ba/Sr', 'Ba + Sr'], ignore_index=False))
# print(melt)

# Calculate means for each sample (messy)
sample_mean = df1.groupby('Sample').mean()
# print(sample_mean.head())

# Another way to calculate means, but need an indexed df to use "level"
#   we need to use a non-indexed dataframe to make our "populations" dataframe
#       so might as well just use one dataframe and the groupby fx to make it simpler
# mean = df.mean(level = 'Sample') # another way to calculate means, but need an indexed df to use "level"

# Create seperate dataframe with sample populations
#   this is so we can merge this with the mean dataframe because the groupby fx just gives sample name and value for each element
populations = df1[['Sample', 'Population']].drop_duplicates('Sample')

# Merge two dataframes
sample_mean = pd.merge(populations, sample_mean, how='right',
                 left_on="Sample", right_on=sample_mean.index)

# Set index
sample_mean = sample_mean.set_index('Sample')


# Drop bad analyses column
# Drop MG samples
sample_mean = sample_mean.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-032', 'ORA-2A-018', 'ORA-2A-035'], axis= 0)
# Drop VCCR samples
sample_mean = sample_mean.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-5B-404A-B'], axis= 0)

sample_mean = sample_mean.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df1['Sample'].value_counts() #can use .size() but that includes NaN values

sample_mean = sample_mean.set_index('Sample')
sample_mean['Count'] = count


# Dataframe Slicing of average values using "isin"
VCCR = sample_mean[sample_mean['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG = sample_mean[sample_mean['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
FG = sample_mean[sample_mean['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP = sample_mean[sample_mean['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-016', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]


# #FGCP = FGCP.drop(['ORA-2A-002'], axis = 0)

# Calculate stdev for each sample (messy)
sample_std = df1.groupby('Sample').std()

# Multiply dataframe by two to get 2 sigma
#sample_std = sample_std *2
# print(sample_std.head())

# Merge two dataframes (stdev and populations)
sample_std = pd.merge(populations, sample_std, how='right',
                      left_on="Sample", right_on=sample_std.index)
#print (sample_std.head())

# Set index
sample_std = sample_std.set_index('Sample')
#print (sample_std.head())

# Drop bad samples
# Drop MG samples
sample_std = sample_std.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-032', 'ORA-2A-018', 'ORA-2A-035'], axis= 0)
# Drop VCCR samples
sample_std = sample_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B','ORA-5B-404A-B'], axis= 0)

# Select sample stdev by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCR_std = sample_std[sample_std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FG_std = sample_std[sample_std['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-002','ORA-2A-016', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

#import csv file
REE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Trace_Avgs_NormalizedREE.xlsx")

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
melt = (REE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
melt = melt.set_index('Sample')

# Dataframe Slicing using "isin"
VCCRREE = melt[melt['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MGREE = melt[melt['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]#VCCRREE = REE.loc[['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,7))


#group plot title
title = fig.suptitle("High-Silica Rhyolite (VCCR + MG) Fiamme Glass Populations", fontsize=18, y = 0.96)

#plot 1 

#create ree plot
plt.subplot(2,2,1)
plot = sns.lineplot(data = MGREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief")
plot1 = sns.lineplot(data = VCCRREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
#plot just populations
plt.legend(h[1:4]+h[5:9],l[1:4]+l[5:9],loc='lower right')

plt.xlabel('')
plt.ylabel('Sample/Chondrite')

#plt.ylim(0.05, 200)

#set y axis to log scale
plot.set(yscale='log')


#plot 2
plt.subplot(2,2,2)
#create trace element plot

# Select elements to plot
x = 'Ba'
y = 'Sr'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]

plot2 = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s',
                       edgecolor="black", s=150, alpha=0.8, legend= False, hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot2 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", marker='^',
                       edgecolor="black", s=150, legend= False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

# Configure legend
h, l = plot.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot 3
plt.subplot(2,2,3)

x = 'U'
y = 'Ti'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]

plot3 = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s',
                       edgecolor="black", s=150, alpha=0.8, legend= False, hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot3 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", marker='^',
                       edgecolor="black", s=150, legend= False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

# Configure legend
h, l = plot.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

#plot 4
plt.subplot(2,2,4)
x = 'Nb'
y = 'Y'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]

plot4 = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s',
                       edgecolor="black", s=150, alpha=0.8, legend= False, hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot4 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", marker='^',
                       edgecolor="black", s=150, legend= False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

# Configure legend
h, l = plot.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

# set size of plot
plt.tight_layout()
#plt.show()

# set size of plot
#sns.set_context("poster")

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/HSR_4Plot_ErrorBars.png', dpi=400)
