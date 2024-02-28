#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

##FROM BA-SR-ALL.PY
# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

# Columns converted to np.float with a dictionary
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# Master spreadsheet with clear mineral analyses removed (excel file!)
df = pd.read_excel(path, sheet_name = 'Data', skiprows = [1], dtype = data, na_values = na_values)

#If Value in included row is equal to zero, drop this row
df = df.loc[~((df['Included'] == 0))]

#Drop included column
df = df.drop('Included', axis = 1)

# drop blank columns
df = df.dropna(axis = 1, how = 'all')
#df = df.dropna(axis=0, how='all')

# NaN treatment:
#   change all negatives and zeroes to NaN
num = df._get_numeric_data()
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

# Change analysis date to a string
s = df['Date']
df['Date'] = (s.dt.strftime('%Y.%m.%d'))

# Dropping Values:
df = df.set_index('Sample')

# #Dropping MG samples because we remeasured these
# df = df.drop(['ORA-2A-036-B','ORA-2A-036','ORA-2A-032','ORA-2A-035'], axis= 0)

# #Dropping VCCR samples because we remeasured these
# df = df.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B'], axis= 0)

# #ADDED THIS RECENTLY
# #Drop VCCR samples because they are the same fiamma:
# df = df.drop(['ORA-5B-405', 'ORA-5B-416'])

# Dropping VCCR samples because we don't have matching SEM values
df = df.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7', 'ORA-5B-412B-CG'], axis= 0)

# # Dropping FG samples because remeasured:
# df = df.drop(['ORA-5B-410','ORA-5B-412B-FG'], axis= 0)
df = df.reset_index()

#---------
# Calculate means for each sample (messy)
# sample_mean = df.groupby('Sample').mean()

sample_mean = df.groupby(
    ['Sample', 'Population', 'Date']).mean()
sample_mean = sample_mean.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df['Sample'].value_counts() #can use .size() but that includes NaN values

sample_mean = sample_mean.set_index('Sample')
sample_mean['Count'] = count

# Calculate stdev for each sample (messy)
sample_std = df.groupby(
    ['Sample', 'Population', 'Date']).std()
sample_std = sample_std.reset_index()

# Add in a column that tells how many samples were calculated for the stdev
sample_std = sample_std.set_index('Sample')
sample_std['Count'] = count

sample_std = sample_std.reset_index()

# Dataframe Slicing of average values using "isin"
VCCR = sample_mean[sample_mean['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG = sample_mean[sample_mean['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
FG = sample_mean[sample_mean['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP = sample_mean[sample_mean['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

# #FGCP = FGCP.drop(['ORA-2A-002'], axis = 0)

sample_mean = sample_mean.reset_index()
#Dropping MG samples because we remeasured these
MG_dupe = sample_mean[sample_mean['Sample'].isin(['ORA-2A-036-B','ORA-2A-036','ORA-2A-032','ORA-2A-035', 'ORA-2A-035-B', 'ORA-2A-032-B', 'ORA-2A-036M'])]

#Dropping VCCR samples because we remeasured these
VCCR_dupe = sample_mean[sample_mean['Sample'].isin(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-5B-416', 'ORA-5B-409', 'ORA-5B-405', 'ORA-5B-406', 'ORA-5B-404A', 'ORA-5B-404A-B'])]

# Dropping FG samples because remeasured:
FG_dupe = sample_mean[sample_mean['Sample'].isin(['ORA-5B-410','ORA-5B-410-B'])]

#ADDED THIS RECENTLY
#Drop VCCR samples because they are the same fiamma:
# df = df.drop(['ORA-5B-405', 'ORA-5B-416'])


# Multiply dataframe by two to get 2 sigma
#sample_std = sample_std *2
# print(sample_std.head())

# Set index
sample_std = sample_std.set_index('Sample')
#print (sample_std.head())

# Select sample stdev by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCR_std = sample_std[sample_std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FG_std = sample_std[sample_std['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-002','ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]


sample_mean = sample_mean.reset_index()
MG_dupe = sample_mean[sample_mean['Sample'].isin(['ORA-2A-032','ORA-2A-035', 'ORA-2A-035-B', 'ORA-2A-032-B'])]

VCCR_dupe = sample_mean[sample_mean['Sample'].isin(['ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-409', 'ORA-5B-406', 'ORA-5B-404A', 'ORA-5B-404A-B'])]

FG_dupe = sample_mean[sample_mean['Sample'].isin(['ORA-5B-410','ORA-5B-410-B'])]

sample_std = sample_std.reset_index()

MG_dupe_std = sample_std[sample_std['Sample'].isin(['ORA-2A-032','ORA-2A-035', 'ORA-2A-035-B', 'ORA-2A-032-B'])]

VCCR_dupe_std = sample_std[sample_std['Sample'].isin(['ORA-5B-406-B','ORA-5B-409-B','ORA-5B-409', 'ORA-5B-406', 'ORA-5B-404A', 'ORA-5B-404A-B'])]

FG_dupe_std = sample_std[sample_std['Sample'].isin(['ORA-5B-410','ORA-5B-410-B'])]

# Plotting
#       Slicing dataframe

# Set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,7))


#group plot title
title = fig.suptitle("Crystal-Rich (VCCR + CG) Fiamme Glass Populations", fontsize=18, y = 0.96)

#plot 1 

#create ree plot
plt.subplot(2,2,1)
x = 'Ba'
y = 'Sr'

xerr1 = MG_dupe_std[x]
yerr1 = MG_dupe_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_dupe_std[x]
yerr2 = VCCR_dupe_std[y]

plot1 = sns.scatterplot(data=MG_dupe, x=x, y=y, hue="Sample", palette="Blues_d", marker='s', style = "Sample",
                       edgecolor="black", s=150, alpha=0.8, legend= False)
plt.errorbar(x=MG_dupe[x], y=MG_dupe[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot1 = sns.scatterplot(data=VCCR_dupe, x=x, y=y, hue="Sample", palette="PuRd_r", style = "Sample",
                       edgecolor="black", s=150, legend= False, alpha=0.8)
plt.errorbar(x=VCCR_dupe[x], y=VCCR_dupe[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

#plot 2
plt.subplot(2,2,2)
#create trace element plot

# Select elements to plot
x = 'U'
y = 'Nd'

xerr1 = MG_dupe_std[x]
yerr1 = MG_dupe_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_dupe_std[x]
yerr2 = VCCR_dupe_std[y]


plot2 = sns.scatterplot(data=MG_dupe, x=x, y=y, hue="Sample", palette="Blues_d", marker='s', style = "Sample",
                       edgecolor="black", s=150, alpha=0.8, legend= False)
plt.errorbar(x=MG_dupe[x], y=MG_dupe[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot2 = sns.scatterplot(data=VCCR_dupe, x=x, y=y, hue="Sample", palette="PuRd_r", style = "Sample",
                       edgecolor="black", s=150, legend= False, alpha=0.8)
plt.errorbar(x=VCCR_dupe[x], y=VCCR_dupe[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

#plot2.text(54,-0.3, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# Configure legend
h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"
# l[1:4] = ('CG 1', 'CG 2', 'CG 3')


plt.legend(h, l, ncol = 1, handlelength = 1, columnspacing = 0.5, loc='center left', bbox_to_anchor=(1, 0.5))


#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot 3
plt.subplot(2,2,3)

x = 'Nb'
y = 'Y'


xerr1 = MG_dupe_std[x]
yerr1 = MG_dupe_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_dupe_std[x]
yerr2 = VCCR_dupe_std[y]


plot3 = sns.scatterplot(data=MG_dupe, x=x, y=y, hue="Sample", palette="Blues_d", marker='s', style = "Sample",
                       edgecolor="black", s=150, alpha=0.8, legend= False)
plt.errorbar(x=MG_dupe[x], y=MG_dupe[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot3 = sns.scatterplot(data=VCCR_dupe, x=x, y=y, hue="Sample", palette="PuRd_r", style = "Sample",
                       edgecolor="black", s=150, legend= False, alpha=0.8)
plt.errorbar(x=VCCR_dupe[x], y=VCCR_dupe[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

#plot3.text(5.1,109, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# Configure legend
h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


# # #plot 4
# plt.subplot(2,2,4)
# x = 'Zr'
# y = 'Ti'

# xerr1 = MG_dupe_std[x]
# yerr1 = MG_dupe_std[y]

# # VCCR Error Bar Values
# xerr2 = VCCR_dupe_std[x]
# yerr2 = VCCR_dupe_std[y]


# plot4 = sns.scatterplot(data=MG_dupe, x=x, y=y, hue="Sample", palette="Blues_d", marker='s', style = "Sample",
#                        edgecolor="black", s=150, alpha=0.8, legend= False)
# plt.errorbar(x=MG_dupe[x], y=MG_dupe[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# plot4 = sns.scatterplot(data=VCCR_dupe, x=x, y=y, hue="Sample", palette="PuRd_r", style = "Sample",
#                        edgecolor="black", s=150, legend= False, alpha=0.8)
# plt.errorbar(x=VCCR_dupe[x], y=VCCR_dupe[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# plt.xlabel(x + ' [ppm]')
# plt.ylabel(y + " [ppm]")

#plot4.text(33,29.53, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')


# Configure legend
#h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)


# l[0] = "Outflow"
# l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


# set size of plot
plt.tight_layout()
#plt.show()

# set size of plot
#sns.set_context("poster")

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/Weird_Laser.png', dpi=800)
