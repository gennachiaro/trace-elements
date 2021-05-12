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
#       Slicing dataframe for averages
#MG = merge.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#MG_index = MG.index

#VCCR = merge.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#VCCR_index = VCCR.index

# Dataframe Slicing of average values using "isin"
VCCR = merge[merge['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG = merge[merge['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
FG = merge[merge['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP = merge[merge['Population'].isin(['ORA-2A-002', 'ORA-2A-016','ORA-2A-003','ORA-2A-023', 'ORA-2A-024'])]

# Drop bad analyses column
MG = MG.drop(['ORA-2A-004', 'ORA-2A-036'], axis = 0)
VCCR = VCCR.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B','ORA-5B-416-B'], axis = 0)

# Plot All Spots
#   Set Dataframe Index
df1 = df1.set_index('Sample')

# Dataframe Slicing of all values using "isin"
VCCR_all = df1[df1['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG_all = df1[df1['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
FG_all = df1[df1['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_all = df1[df1['Population'].isin(['ORA-2A-002', 'ORA-2A-016','ORA-2A-003','ORA-2A-023', 'ORA-2A-024'])]

# Drop bad analyses column
MG_all = MG_all.drop(['ORA-2A-004', 'ORA-2A-036'], axis = 0)

# Dataframe slicing using sample names
#VCCR_all = df1.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#MG_all = df1.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]

#create trace element plot
#plot = sns.scatterplot(data = VCCR_all, x= x, y= y,hue = "Population", palette= "gray", edgecolor="black", marker = '^', s=150, style = "Population", alpha = 0.2, legend = False, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
#plot = sns.scatterplot(data = MG_all, x= x, y= y,hue = "Population" , palette="gray",marker = 's', edgecolor="black", s=150, alpha = 0.2,style = "Population", legend = False, hue_order = ['MG 1', 'MG 2', 'MG 3'])

# Set background color
sns.set_style("darkgrid")

# Set axes limits
#plt.ylim(10, 50)
#plt.xlim (-0.2,0.4)

# Set color palette
#sns.set_palette("PuBuGn_d")

# Get error bar values

# Select MG standard samples by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
# Drop bad samples
MG_std = MG_std.drop(['ORA-2A-004', 'ORA-2A-036'], axis = 0)

# Select VCCR standard samples by population
VCCR_std = sample_std[sample_std['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
# Drop bad samples
VCCR_std = VCCR_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B','ORA-5B-416-B'], axis = 0)

# Select FG standard samples by population
FG_std = sample_std[sample_std['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
# Drop bad samples
#FG_std = FG_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B','ORA-5B-416-B'], axis = 0)

# Select FGCP standard samples by population
FGCP_std = sample_std[sample_std['Population'].isin(['ORA-2A-002', 'ORA-2A-016','ORA-2A-003','ORA-2A-023', 'ORA-2A-024'])]
# Drop bad samples
#FGCP_std = FGCP_std.drop(['ORA-2A-002', 'ORA-2A-016','ORA-2A-003','ORA-2A-023', 'ORA-2A-024'], axis = 0)


#import csv file for major elements
df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/ora-major-elements.csv', index_col=0)

MGm = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035', 'ORA-2A-040']]
MGm_index = MGm.index

VCCRm = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
VCCRm_index = VCCRm.index

#fine grained 
df2 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/majors-fgcp-grouped.csv', index_col=0)

FGCPm = df2.loc[['ORA-2A-002','ORA-2A-003','ORA-2A-016','ORA-2A-023','ORA-2A-024']]
FGCPm_index = FGCPm.index

FGm = df2.loc [['ORA-5B-414','ORA-5B-410','ORA-5B-412A','ORA-5B-412B']]
FGm_index = FGm.index

#import csv file for trace elements
#tr = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_All.csv', index_col=1)

#FGCP = tr.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3','ORA-2A-016-Type1', 'ORA-2A-016-Type2', 'ORA-2A-016-Type3', 'ORA-2A-016-Type4','ORA-2A-003','ORA-2A-023','ORA-2A-024-TYPE1','ORA-2A-024-TYPE2','ORA-2A-024-TYPE3','ORA-2A-024-TYPE4']]   
#FGCP_index = FGCP.index

#MG = tr.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#MG_index = MG.index

#VCCR = tr.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
#VCCR_index = VCCR.index

#FG = tr.loc [['ORA-5B-410-FG','ORA-5B-412-FG','ORA-5B-414-FG']]
#FG_index = FG.index

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,7))

#group plot title
title = fig.suptitle("All Ora Fiamme Glass", fontsize=16, y = 0.925)

#plot 1 

#create major element plot
plt.subplot(2,2,1)
plot = sns.scatterplot(data = FGCPm, x= 'SiO2', y='K2O',hue = "Population", style = "Population", palette="Greens_r",legend="brief", markers = ('o','s','X','P','D'),edgecolor="black", s=150, alpha = 0.5, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plot = sns.scatterplot(data = FGm, x= 'SiO2', y='K2O',hue = "Population", style = "Population", palette="OrRd_r",legend="brief", markers = ('s', '^', 'X'), edgecolor="black", alpha = 0.5, s=150, hue_order = ['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
plot = sns.scatterplot(data = MGm, x= 'SiO2', y='K2O',hue = "Population" , style = MGm_index, marker = '^', palette="Blues_r", edgecolor="black", s=150,alpha = 0.5,legend="brief")
plot = sns.scatterplot(data = VCCRm, x= 'SiO2', y='K2O',hue = "Population", style = VCCRm_index, marker = 's',palette="PuRd_r", edgecolor="black", s=150, legend = "brief", alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

plt.xticks(range(64, 82, 2))
plt.ylim(0.2, 4.9)

#set location of legend

h,l = plot.get_legend_handles_labels()
plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26],l[1:7]+l[7:10]+l[11:14]+l[23:26],loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize = 11)

#plot 2
plt.subplot(2,2,2)
#create trace element plot

# Plotting
# Select elements to plot
x = 'Ba'
y = 'Sr'

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

# Create plot
#   All one symbol
plot = sns.scatterplot(data = MG, x= x, y= y, hue = "Population", palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.8, legend = False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x = MG[x], y = MG[y], xerr = xerr1, yerr = yerr1, ls = 'none', ecolor = 'cornflowerblue', elinewidth = 1, capsize = 2, alpha = 0.8)

plot = sns.scatterplot(data = VCCR, x= x, y=y, hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, alpha = 0.8, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x = VCCR[x], y = VCCR[y], xerr = xerr2, yerr = yerr2, ls = 'none', ecolor = 'palevioletred', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

plot = sns.scatterplot(data = FGCP, x= x, y=y, hue = "Population", palette="Greens_r", style = "Population", edgecolor="black", s=150, legend = False, alpha = 0.8, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plt.errorbar(x = FGCP[x], y = FGCP[y], xerr = xerr3, yerr = yerr3, ls = 'none', ecolor = 'green', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

plot = sns.scatterplot(data = FG, x= x, y=y, hue = "Population", palette="OrRd_r", style = 'Population', edgecolor="black", s=150, legend = False, alpha = 0.8, markers=('^', 'X', 's'), hue_order = ['ORA-5B-412','ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x = FG[x], y = FG[y], xerr = xerr4, yerr = yerr4, ls = 'none', ecolor = 'orange', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

#plt.ylim(-60, 590)
#plt.xlim (-40,510)

#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot 3
plt.subplot(2,2,3)

# Plotting
# Select elements to plot
x = 'Ba'
y = 'Y'

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

# Create plot
#   All one symbol
plot = sns.scatterplot(data = MG, x= x, y= y, hue = "Population", palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.8, legend = False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x = MG[x], y = MG[y], xerr = xerr1, yerr = yerr1, ls = 'none', ecolor = 'cornflowerblue', elinewidth = 1, capsize = 2, alpha = 0.8)

plot = sns.scatterplot(data = VCCR, x= x, y=y, hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, alpha = 0.8, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x = VCCR[x], y = VCCR[y], xerr = xerr2, yerr = yerr2, ls = 'none', ecolor = 'palevioletred', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

plot = sns.scatterplot(data = FGCP, x= x, y=y, hue = "Population", palette="Greens_r", style = "Population", edgecolor="black", s=150, legend = False, alpha = 0.8, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plt.errorbar(x = FGCP[x], y = FGCP[y], xerr = xerr3, yerr = yerr3, ls = 'none', ecolor = 'green', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

plot = sns.scatterplot(data = FG, x= x, y=y, hue = "Population", palette="OrRd_r", style = 'Population', edgecolor="black", s=150, legend = False, alpha = 0.8, markers=('^', 'X', 's'), hue_order = ['ORA-5B-412','ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x = FG[x], y = FG[y], xerr = xerr4, yerr = yerr4, ls = 'none', ecolor = 'orange', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

#plt.ylim(10,125)
#plt.xlim (-40,510)


#plot 4
plt.subplot(2,2,4)
# Plotting
# Select elements to plot
x = 'Nb'
y = 'Rb'

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

# Create plot
#   All one symbol
plot = sns.scatterplot(data = MG, x= x, y= y, hue = "Population", palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.8, legend = False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x = MG[x], y = MG[y], xerr = xerr1, yerr = yerr1, ls = 'none', ecolor = 'cornflowerblue', elinewidth = 1, capsize = 2, alpha = 0.8)

plot = sns.scatterplot(data = VCCR, x= x, y=y, hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, alpha = 0.8, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x = VCCR[x], y = VCCR[y], xerr = xerr2, yerr = yerr2, ls = 'none', ecolor = 'palevioletred', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

plot = sns.scatterplot(data = FGCP, x= x, y=y, hue = "Population", palette="Greens_r", style = "Population", edgecolor="black", s=150, legend = False, alpha = 0.8, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plt.errorbar(x = FGCP[x], y = FGCP[y], xerr = xerr3, yerr = yerr3, ls = 'none', ecolor = 'green', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

plot = sns.scatterplot(data = FG, x= x, y=y, hue = "Population", palette="OrRd_r", style = 'Population', edgecolor="black", s=150, legend = False, alpha = 0.8, markers=('^', 'X', 's'), hue_order = ['ORA-5B-412','ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x = FG[x], y = FG[y], xerr = xerr4, yerr = yerr4, ls = 'none', ecolor = 'orange', elinewidth = 1, capsize = 2, barsabove = False, alpha = 0.8)

#plt.ylim(170,780)
#plt.xlim (8.1,50.8)

# set size of plot
#plt.subplots_adjust(hspace = 0.25, wspace = 0.2)
#plt.tight_layout(pad= 1.0)
#plt.show()

# set size of plot
#sns.set_context("poster")