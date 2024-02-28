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
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

#USE MYENV

##FROM BA-SR-ALL.PY
# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows


# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****', 'nan']

# Columns converted to np.float with a dictionary
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# Master spreadsheet with clear mineral analyses removed (excel file!)
df = pd.read_excel(path, sheet_name = 'Data', skiprows = [1], dtype = data, na_values = na_values)

#If Value in included row is equal to zero, drop this row
df = df.loc[~((df['Included'] == 0))]

#Drop included column
df = df.drop('Included', axis = 1)

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
df = df.dropna(axis=1, how='all')

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

#Dropping MG samples because we remeasured these
df = df.drop(['ORA-2A-036-B','ORA-2A-036','ORA-2A-032','ORA-2A-035'], axis= 0)

#Dropping VCCR samples because we remeasured these
df = df.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-5B-404A-B'], axis= 0)

# Dropping VCCR samples because we don't have matching SEM values
df = df.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7', 'ORA-5B-412B-CG'], axis= 0)
df = df.reset_index()

# Get All Spots for sample 
# Dataframe Slicing of average values using "isin"
all_2A_002 = df[df['Sample'].isin(['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3'])]

all_2A_024 = df[df['Sample'].isin(['ORA-2A-024-TYPE1','ORA-2A-024-TYPE2','ORA-2A-024-TYPE3','ORA-2A-024-TYPE4'])]

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

sample_mean = sample_mean.reset_index()

# Dataframe Slicing of average values using "isin"
VCCR = sample_mean[sample_mean['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG = sample_mean[sample_mean['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
FG = sample_mean[sample_mean['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP = sample_mean[sample_mean['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-016', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

ORA2A024 = sample_mean[sample_mean['Population'].isin(
    ['ORA-2A-024'])]

ORA2A002 = sample_mean[sample_mean['Population'].isin(
    ['ORA-2A-002'])]

# Multiply dataframe by two to get 2 sigma
#sample_std = sample_std *2
# print(sample_std.head())

# Set index
#sample_std = sample_std.set_index('Sample')
#print (sample_std.head())

# Select sample stdev by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCR_std = sample_std[sample_std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FG_std = sample_std[sample_std['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-002','ORA-2A-016', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

ORA2A024_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-024'])]

ORA2A002_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-002'])]

# Add in REE
#import xlsx file
REE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Trace_Avgs_NormalizedREE.xlsx", sheet_name = 'FGCP_Normalized')

#na_values = ['nan']

#  change all negatives to NaN
num = REE._get_numeric_data()
num[num <= 0] = np.nan

#REE = REE.dropna(axis=1, how = 'all')

REE = REE.dropna(axis=1, how = 'all')

REE = REE.dropna(axis=0, how = 'any')


#REE = REE.dropna(axis=0, how = 'any')

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
melt = (REE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
#melt = melt.set_index('Sample')

#melt = melt.set_index('Sample')
#melt = melt.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG'], axis= 0)
#melt = melt.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)
#melt = melt.reset_index()

# Dataframe Slicing using "isin"
ORA_002_REE = melt[melt['Population'].isin(['ORA-2A-002'])]
ORA_024_REE = melt[melt['Population'].isin(['ORA-2A-024'])]



#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(11,9))

# group plot title
#title = fig.suptitle("All Ora Fiamme Glass Trace Elements", fontsize=16, y=0.925)

#plot 1 


# Plotting
# Select elements to plot
x = 'La/Lu'
y = 'Eu/Eu*'

# x = 'Ba'
# y = 'Nb'

# 2A 024 Error Bar Values
xerr1 = ORA2A002_std[x]
yerr1 = ORA2A002_std[y]

# 2A 002 Error Bar Values
xerr2 = ORA2A024_std[x]
yerr2 = ORA2A024_std[y]


# Create plot
plt.subplot(2,2,1)

plt.title("ORA-2A-002", fontsize=18.5, fontweight=0, color='black', y = 0.99)

# Show all symbols
ORA2A002 = ORA2A002.replace(regex={'ORA-2A-002-Type1': 'ORA-2A-002-Type 1', 'ORA-2A-002-Type2': 'ORA-2A-002-Type 2', 'ORA-2A-002-Type3': 'ORA-2A-002-Type 3'})

plot = sns.scatterplot(data = all_2A_002, x= x, y=y, hue = "Sample", style = "Sample", palette="gray", edgecolor="black", s=200, alpha = 0.2, legend=False, markers = ['o','X','s'], hue_order=['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3'])


plot = sns.scatterplot(data=ORA2A002, x=x, y=y, hue="Sample", palette="Greens_r", style="Sample", edgecolor="black",
                       s=250, legend='brief', alpha=0.85,  markers = ['o','s','X'])
plt.errorbar(x=ORA2A002[x], y=ORA2A002[y], xerr=xerr1, yerr=yerr1, ls='none', ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

#Y vs. U
# plot.text(14.3,53, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#Y vs. Gd
#plot.text(14.9,53, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

plt.xlabel(x, fontsize = 18.5)
plt.ylabel(y, fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

from scipy import stats
import numpy as np
res = stats.linregress(ORA2A002[x], ORA2A002[y])
print(f"R-squared: {res.rvalue**2:.6f}")


h, l = plot.get_legend_handles_labels()
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', fontsize = 18.5, ncol = 1, handlelength = 1.5, markerscale = 2)

# plt.xlabel(x + ' [ppm]')
# plt.ylabel(y + " [ppm]")

# plt.ylim((51, 120.2) )
# plt.xlim (6.4, 19)

plt.ylim((-0.0025,0.067) )
plt.xlim (8, 44)

#plot 2
plt.subplot(2,2,2)
#create trace element plot
plt.title("ORA-2A-024", fontsize=18.5, fontweight=0, color='black', y = 0.99)

# Show all symbols
ORA2A024 = ORA2A024.replace(regex={'ORA-2A-024-TYPE1': 'ORA-2A-024-Type 1','ORA-2A-024-TYPE2': 'ORA-2A-024-Type 2' ,'ORA-2A-024-TYPE3': 'ORA-2A-024-Type 3','ORA-2A-024-TYPE4': 'ORA-2A-024-Type 4'})

plot2 = sns.scatterplot(data = all_2A_024, x= x, y=y, hue = "Sample", style = "Sample", palette="gray", edgecolor="black", s=200, alpha = 0.2, legend=False, ci = 99.7,  linewidth = 1)



plot2 = sns.scatterplot(data=ORA2A024, x=x, y=y, hue="Sample", palette="Greens_r", style="Sample", edgecolor="black",
                       s=250, legend='brief', alpha=0.85)
plt.errorbar(x=ORA2A024[x], y=ORA2A024[y], xerr=xerr2, yerr=yerr2, ls='none', ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# plot = sns.scatterplot(data=FGCP, x=x, y=y, hue="Population", palette="Greens_r", style="Population", edgecolor="black",
#                        s=150, legend=False, alpha=0.8, hue_order=['ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
# plt.errorbar(x=FGCP[x], y=FGCP[y], xerr=xerr3, yerr=yerr3, ls='none',
#              ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# plot = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
#                        s=150, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
# plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
#              ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# plt.xlabel(x + ' [ppm]')
# plt.ylabel(y + " [ppm]")

# plt.ylim((51, 120.2) )
# plt.xlim (6.4, 19)

# plot2.text(28.5,-0.0007, str('error bars $\pm$ 1$\sigma$'), fontsize = 18.5, fontweight = 'normal')

from scipy import stats
import numpy as np
res = stats.linregress(ORA2A024[x], ORA2A024[y])
print(f"R-squared: {res.rvalue**2:.6f}")


plt.xlabel(x, fontsize = 18.5)
plt.ylabel(y, fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.ylim((-0.0025,0.067) )
plt.xlim (8, 44)

#plot2.text(15.7,67.4, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

h, l = plot2.get_legend_handles_labels()
# Legend inside of plot
plt.legend(h[1:5]+h[5:8], l[1:5]+l[5:8], loc='upper left', ncol=1, handlelength = 1, columnspacing = 0.5, fontsize = 18.5, markerscale = 2)

# ---------
#plot 3
plt.subplot(2,2,3)

ORA_002_REE = ORA_002_REE.replace(regex={'ORA-2A-002-Type1': 'ORA-2A-002-Type 1', 'ORA-2A-002-Type2': 'ORA-2A-002-Type 2', 'ORA-2A-002-Type3': 'ORA-2A-002-Type 3'})

error_config = {'alpha' : 0.3}
plot = sns.lineplot(data = ORA_002_REE, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens_d",legend="brief", hue_order = ('ORA-2A-002-Type 1', 'ORA-2A-002-Type 2','ORA-2A-002-Type 3'), ci = 'sd', err_kws = error_config)

#palette1 = []
#plot = sns.lineplot(data = ORA_002_REE, x= 'variable', y='value', sort = False, palette="Greens_d",legend="brief")

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='lower right', fontsize = 18.5, ncol = 1, handlelength = 1, markerscale = 10, columnspacing = 0.5)

plt.xlabel('')
plt.ylabel('Sample/Chondrite', fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=18.5)

#plt.text(8.1,0.12, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')
plt.text(-0.2,0.12, str('error envelopes $\pm$ 1 std'), fontsize = 18.5, fontweight = 'normal')

plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)

#set y axis to log scale
plot.set(yscale='log')
plt.ylim( (10**-1,10**2.2) )

for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

for tick in plot.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(25)

#plt.yscale('log')


# -----------
#plot 4
plt.subplot(2,2,4)

ORA_024_REE = ORA_024_REE.replace(regex={'ORA-2A-024-TYPE1': 'ORA-2A-024-Type 1','ORA-2A-024-TYPE2': 'ORA-2A-024-Type 2' ,'ORA-2A-024-TYPE3': 'ORA-2A-024-Type 3','ORA-2A-024-TYPE4': 'ORA-2A-024-Type 4'})

plot = sns.lineplot(data = ORA_024_REE, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens_d",legend="brief", ci = 'sd', err_kws = error_config,  linewidth = 1)

#plot = sns.lineplot(data = ORA_024_REE, x= 'variable', y='value', sort = False, palette="Greens_d",legend="brief")

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
# Legend inside of plot
plt.legend(h[1:5]+h[5:8], l[1:5]+l[5:8], loc='lower right', fontsize = 18.5, ncol = 1, handlelength = 1, markerscale = 10, columnspacing = 0.5)


plt.xlabel('')
plt.ylabel('Sample/Chondrite', fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=18.5)

plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)

#plot1.text(-0.5,0.45, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#plt.set_ylim(-10, 1200)

#set y axis to log scale
#plt.set_ylim (-10, 120)

plt.yscale('log')
plt.ylim( (10**-1,10**2.2) )
#plt.ylim (-10, 120)

for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

for tick in plot.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(25)

#sns.set_context("paper") 
#plt.tight_layout()

plt.tight_layout(pad = 1)

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/FGCP_4plot_REE_sd_eu-eu*log_numbers_Revision2_Errorbars.svg', dpi=800)

