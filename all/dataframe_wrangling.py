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
#path = os.path.normcase(path) # This changes path to be compatible with windows

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

# If Value in included row is equal to zero, drop this row
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
all = (df1.melt(id_vars=['Sample','Spot','Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))

# Calculate means for each sample (messy)
sample_mean = df1.groupby(
    ['Sample', 'Population', 'Date']).mean()
sample_mean = sample_mean.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df1['Sample'].value_counts() #can use .size() but that includes NaN values

sample_mean = sample_mean.set_index('Sample')
sample_mean['Count'] = count
#sample_mean = sample_mean.reset_index()

# Drop MG samples
sample_mean = sample_mean.drop(['ORA-2A-004', 'ORA-2A-036'], axis= 0)
# Drop VCCR samples
sample_mean = sample_mean.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B'], axis= 0)


# Set indexes
#sample_mean = sample_mean.set_index('Sample')

# Calculate stdev for each sample (messy)
sample_std = df1.groupby(
    ['Sample', 'Population', 'Date']).std()
sample_std = sample_std.reset_index()

# Add in a column that tells how many samples were calculated for the stdev
sample_std = sample_std.set_index('Sample')

sample_std['Count'] = count

# Drop MG samples
sample_std = sample_std.drop(['ORA-2A-004', 'ORA-2A-036'], axis= 0)

# Drop VCCR samples
sample_std = sample_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B'], axis= 0)

#sample_std = sample_std.reset_index()

# Set index
sample_std = sample_std.set_index('Sample')

#-------------------------
# Slicing dataframe: Means

# # Dataframe Slicing of average values using "isin"
VCCR = sample_mean[sample_mean['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG = sample_mean[sample_mean['Population'].isin(
    ['MG 1', 'MG 2', 'MG 3'])]



# FG = sample_mean[sample_mean['Population'].isin(
#     ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
# FGCP = sample_mean[sample_mean['Population'].isin(
#     ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])

# Set Indices
#MG = MG.reset_index()
#MG = MG.set_index('Sample')
#VCCR = VCCR.set_index('Sample')


# Drop bad analyses 
#MG = MG.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-001'], axis=0)

# MG = MG.drop(['ORA-2A-004', 'ORA-2A-036'], axis= 0)

# VCCR = VCCR.drop(['ORA-5B-405-B', 'ORA-5B-406-B',
#                  'ORA-5B-409-B', 'ORA-5B-416-B'], axis=0)
#FGCP = FGCP.drop(['ORA-2A-002'], axis = 0)

# Slicing dataframe: StDevs

# # Dataframe Slicing of stdev values using "isin"
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCR_std = sample_std[sample_std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]

   
FG_std = sample_std[sample_std['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])

# Drop bad analyses
# Set Indices
#MG_std = MG_std.set_index('Sample')
#VCCR_std = VCCR_std.set_index('Sample')


#MG_std = MG_std.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-001'], axis=0)
# MG_std = MG_std.drop(['ORA-2A-004', 'ORA-2A-036'], axis=0)

# VCCR_std = VCCR_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B'], axis=0)

#FGCP_std = FGCP_std.drop(
#   ['ORA-2A-002'], axis = 0)
#FG_std = FG_std.drop(
#['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B','ORA-5B-416-B'], axis = 0)

#--------------------
# Plotting

# Select elements to plot
x = "Ba"
y = "Sr"

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]

# FGCP Error Bar Values
xerr3 = FGCP_std[x]
yerr3 = FGCP_std[y]

# FG Error Bar Values
xerr4 = FG_std[x]
yerr4 = FG_std[y]

# Set background color
sns.set_style("darkgrid")

# Create plot
#   All one symbol
plot = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s',
                       edgecolor="black", s=150, alpha=0.8, legend=False, hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", marker='^',
                       edgecolor="black", s=150, legend=False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plot = sns.scatterplot(data=FGCP, x=x, y=y, hue="Population", palette="Greens_r", style="Population", edgecolor="black",
                       s=150, legend=False, alpha=0.8, hue_order=['ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plt.errorbar(x=FGCP[x], y=FGCP[y], xerr=xerr3, yerr=yerr3, ls='none',
             ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plot = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
                       s=150, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
             ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

#  Different symbol for each sample
#plot = sns.scatterplot(data = VCCR, x= 'Sr', y='Ba',hue = "Population", style = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = "brief", alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

#plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)

# Set y axis to log scale
# plot.set(yscale='log')
# plot.set(xscale='log')

# Configure legend
h, l = plot.get_legend_handles_labels()

# Set location of legend

#plt.legend(loc='upper left')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

# Populations
#plt.legend(h[1:4]+h[13:16],l[1:4]+l[13:16],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Samples + populations
#plt.legend(h[0:4]+ h[5:12]+h[13:16]+h[17:27],l[0:4]+l[5:12]+ l[13:16]+l[17:27],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# General title
plt.suptitle("All Fiamme Glass", fontsize=15,
             fontweight=0, color='black', y=0.95)

# Set size of plot
sns.set_context("paper")

plt.figure(figsize=(18, 12), dpi=400)

plt.show()

#plt.savefig("myplot.png", dpi = 400)