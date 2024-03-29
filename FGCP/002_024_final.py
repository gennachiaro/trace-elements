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


# #FGCP = FGCP.drop(['ORA-2A-002'], axis = 0)

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

# Plotting
#       Slicing dataframe

# Set background color
sns.set_style("darkgrid")

# Set axes limits
#plt.ylim(10, 50)
#plt.xlim (-0.2,0.4)

# Set color palette
# sns.set_palette("PuBuGn_d")

# Plotting
# Select elements to plot
x = 'Nb'
y = 'Y'

# x = 'Ba'
# y = 'Sr'

# 2A 024 Error Bar Values
xerr1 = ORA2A002_std[x]
yerr1 = ORA2A002_std[y]

# 2A 002 Error Bar Values
xerr2 = ORA2A024_std[x]
yerr2 = ORA2A024_std[y]


#plot matrix
fig = plt.figure(figsize=(10,4))


# Create plot
#   All one symbol

#plot 1
plt.subplot(1,2,1)
plt.title("ORA-2A-002", fontsize=13.5, fontweight=0, color='black', y = 0.99)


# Show all symbols
plot = sns.scatterplot(data = all_2A_002, x= x, y=y, hue = "Sample", style = "Sample", palette="gray", edgecolor="black", s=150, alpha = 0.2, legend=False, markers = ['o','X','s'], hue_order=['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3'])

plt.ylim(50, 120)
#plt.xlim(6, 19)
plt.xlim(20, 50)

ORA2A002 = ORA2A002.replace(regex={'ORA-2A-002-Type1': 'Type 1', 'ORA-2A-002-Type2': 'Type 2', 'ORA-2A-002-Type3': 'Type 3'})

plot = sns.scatterplot(data=ORA2A002, x=x, y=y, hue="Sample", palette="Greens_r", style="Sample", edgecolor="black",
                       s=200, legend='brief', alpha=0.85,  markers = ['o','s','X'])
plt.errorbar(x=ORA2A002[x], y=ORA2A002[y], xerr=xerr1, yerr=yerr1, ls='none', ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

#Y vs. U
# plot.text(14.3,53, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#Y vs. Gd
#plot.text(14.3,53, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')


h, l = plot.get_legend_handles_labels()
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

#plot 2
plt.subplot(1,2,2)
plt.title("ORA-2A-024", fontsize=13.5, fontweight=0, color='black', y = 0.99)

# Show all symbols
plot2 = sns.scatterplot(data = all_2A_024, x= x, y=y, hue = "Sample", style = "Sample", palette="gray", edgecolor="black", s=150, alpha = 0.2, legend=False)

ORA2A024 = ORA2A024.replace(regex={'ORA-2A-024-TYPE1': 'Type 1','ORA-2A-024-TYPE2': 'Type 2' ,'ORA-2A-024-TYPE3': 'Type 3','ORA-2A-024-TYPE4': 'Type 4'})

plot2 = sns.scatterplot(data=ORA2A024, x=x, y=y, hue="Sample", palette="Greens_r", style="Sample", edgecolor="black",
                       s=200, legend='brief', alpha=0.85)
plt.errorbar(x=ORA2A024[x], y=ORA2A024[y], xerr=xerr2, yerr=yerr2, ls='none', ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.ylim(50, 120)
plt.xlim(20, 50)
# plot = sns.scatterplot(data=FGCP, x=x, y=y, hue="Population", palette="Greens_r", style="Population", edgecolor="black",
#                        s=150, legend=False, alpha=0.8, hue_order=['ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
# plt.errorbar(x=FGCP[x], y=FGCP[y], xerr=xerr3, yerr=yerr3, ls='none',
#              ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# plot = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
#                        s=150, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
# plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
#              ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

#Y vs. U
# plot2.text(15.7,67.4, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#Y vs. Gd
#plot2.text(15.7,67.4, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')


h, l = plot2.get_legend_handles_labels()
# Legend inside of plot
plt.legend(h[1:5]+h[5:8], l[1:5]+l[5:8], loc='best', ncol=1)


#   Different symbol for each population
#plot = sns.scatterplot(data = VCCR, x= 'Sr', y='Ba',hue = "Population", style = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = "brief", alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

#plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)

# Set y axis to log scale
# plt.yscale('log')
# plt.xscale('log')


# plot.set(yscale='log')
# plot.set(xscale='log')

# Set location of legend
#plt.legend(loc='upper left')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Configure legend
h, l = plot.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Populations
#plt.legend(h[1:4]+h[13:16],l[1:4]+l[13:16],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Samples + populations
#plt.legend(h[0:4]+ h[5:12]+h[13:16]+h[17:27],l[0:4]+l[5:12]+ l[13:16]+l[17:27],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# General title
# plt.suptitle("High-Silica Rhyolite (MG + VCCR) Fiamme Glass", fontsize=15,
#              fontweight=0, color='black', y=0.95)
plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/2A-002-024_Y-Nd.png', dpi=600)

# Set size of plot
sns.set_context("paper") 

#plt.figure(figsize=(18, 12), dpi=400)

#plt.show()

#plt.show()

#Write summary statistics to excel sheet

# with pd.ExcelWriter("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/stats/All_Trace_Corrected_Stats_Final.xlsx") as writer:
#     MG.to_excel(writer, sheet_name = "MG")
#     VCCR.to_excel(writer, sheet_name = "VCCR")
#     FG.to_excel(writer, sheet_name = "FG")
#     FGCP.to_excel(writer, sheet_name = "FGCP")
#     MG_std.to_excel(writer, sheet_name = "MG_std")
#     VCCR_std.to_excel(writer, sheet_name = "VCCR_std")
#     FG_std.to_excel(writer, sheet_name = "FG_std")
#     FGCP_std.to_excel(writer, sheet_name = "FGCP_std")