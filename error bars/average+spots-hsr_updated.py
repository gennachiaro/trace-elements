#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:47:20 2021

@author: gennachiaro
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib as mpl

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

df1['Gd/Lu'] = (df1['Gd']/df1['Lu'])

df1['La/Yb'] = (df1['La']/df1['Yb'])

df1['Yb/Dy'] = (df1['Yb']/df1['Dy'])



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
sample_mean = sample_mean.drop(['ORA-2A-036', 'ORA-2A-032', 'ORA-2A-035'], axis= 0)
# Drop VCCR samples
sample_mean = sample_mean.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-5B-404A-B'], axis= 0)
# Drop to match with SEM-Data!
sample_mean = sample_mean.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG'], axis= 0)

#Drop because measured two of the same fiamme!
sample_mean = sample_mean.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)

#Dropping FG samples because we remeasured these
sample_mean = sample_mean.drop([ 'ORA-5B-410','ORA-5B-412B-FG'], axis= 0)



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
    ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]


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
sample_std = sample_std.drop(['ORA-2A-032', 'ORA-2A-035'], axis= 0)
# Drop VCCR samples
#sample_std = sample_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B','ORA-5B-404A-B'], axis= 0)
# Drop to match with SEM-Data!
sample_std = sample_std.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG', 'ORA-5B-410','ORA-5B-412B-FG'], axis= 0)

#Drop because measured two of the same fiamme!
sample_std = sample_std.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)


# Select sample stdev by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCR_std = sample_std[sample_std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FG_std = sample_std[sample_std['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]


# Plot All Spots
#   Set Dataframe Index
df1 = df1.set_index('Sample')

# Dataframe Slicing using "isin"
VCCR_all = df1[df1['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG_all = df1[df1['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]

# Drop bad analyses column
MG_all = MG_all.drop(['ORA-2A-032', 'ORA-2A-035', 'ORA-2A-036'], axis=0)

VCCR_all = VCCR_all.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-5B-404A-B'], axis= 0)

#Drop because measured two of the same fiamme!
VCCR_all = VCCR_all.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)



# Plotting
# Select elements to plot
x = 'Ba'
y = 'Sr'

# create trace element plot
plot = sns.scatterplot(data=VCCR_all, x=x, y=y, hue="Population", palette="gray", edgecolor="black",
                       marker='^', s=150, alpha=0.2, legend=False, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data=MG_all, x=x, y=y, hue="Population", palette="gray", marker='s', edgecolor="black",
                       s=150, alpha=0.2, legend=False, hue_order=['MG 1', 'MG 2', 'MG 3'])

# Set background color
sns.set_style("darkgrid")

# Set axes limits
#plt.ylim(10, 50)
#plt.xlim (-0.2,0.4)

# Set color palette
# sns.set_palette("PuBuGn_d")

# Get error bar values

# Select MG standard samples by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
# Drop bad samples
MG_std = MG_std.drop(['ORA-2A-004', 'ORA-2A-036'], axis=0)

# Select VCCR standard samples by population
VCCR_std = sample_std[sample_std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
# Drop bad samples
VCCR_std = VCCR_std.drop(
    ['ORA-5B-405-B', 'ORA-5B-406-B', 'ORA-5B-409-B', 'ORA-5B-416-B'], axis=0)

# MG Error Bar Values
#xerr1 = MG_std['Sr']
#yerr1 = MG_std['Ba']

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]

# Create plot
#   All one symbol
plot = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d",
                       marker='s', edgecolor="black", s=150, alpha=0.8, legend="brief")
# Plot Error bars
#plt.errorbar(data = MG, x = 'Sr', y = 'Ba', xerr = xerr1, yerr = yerr1)
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

#   Different symbol for each population
#plot = sns.scatterplot(data = MG, x= 'Sr', y= 'Ba',hue = "Population" , style = MG.index, palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.5, legend = "brief")

# Seperated based on types
#plot = sns.scatterplot(data = VCCR1, x= 'Sr', y='Ba',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, style = VCCR1_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], legend = "brief")

# Population based to fit all the types
#   All one symbol
plot = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", marker='^',
                       edgecolor="black", s=150, legend="brief", alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
#plt.errorbar(data = VCCR, x = 'Sr', y = 'Ba', xerr = xerr2, yerr = yerr2)

plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


#   Different symbol for each population
#plot = sns.scatterplot(data = VCCR, x= 'Sr', y='Ba',hue = "Population", style = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = "brief", alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

#plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)

# Set y axis to log scale
# plot.set(yscale='log')
# plot.set(xscale='log')

# Set location of legend
#plt.legend(loc='upper left')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Configure legend
h, l = plot.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# Populations
#plt.legend(h[1:4]+h[13:16],l[1:4]+l[13:16],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Samples + populations
#plt.legend(h[0:4]+ h[5:12]+h[13:16]+h[17:27],l[0:4]+l[5:12]+ l[13:16]+l[17:27],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# General title
plt.suptitle("High-Silica Rhyolite (MG + VCCR) Fiamme Glass",
             fontsize=15, fontweight=0, color='black', y=0.95)

# Set size of plot
sns.set_context("paper")

plt.show()
