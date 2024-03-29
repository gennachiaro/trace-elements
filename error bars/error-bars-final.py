#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

##FROM BA-SR-ALL.PY

# Import csv file
# All values
#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64}, na_values= na_values)

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

#Drop included column
df1 = df1.drop('Included', axis = 1)

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

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
melt = (df1.melt(id_vars=['Sample', 'Spot', 'Population'], value_vars=['Li', 'Mg', 'Al', 'Si', 'Ca', 'Sc', 'Ti', 'Ti.1', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
        'Tb', 'Gd.1', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U', 'Rb/Sr', 'Ba/Y', 'Zr/Y', 'Zr/Ce', 'Zr/Nb', 'U/Ce', 'Ce/Th', 'Rb/Th', 'Th/Nb', 'U/Y', 'Sr/Nb', 'Gd/Yb', 'U/Yb', 'Zr/Hf', 'Ba/Sr', 'Ba + Sr'], ignore_index=False))
# print(melt)

# Calculate means for each sample (messy)
sample_mean = df1.groupby('Sample').mean()

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
#sample_mean = sample_mean.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-032','ORA-2A-035', 'ORA-2A-018'], axis= 0)
#sample_mean = sample_mean.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-032','ORA-2A-035'], axis= 0)
sample_mean = sample_mean.drop(['ORA-2A-036', 'ORA-2A-032','ORA-2A-035'], axis= 0)

#sample_mean = sample_mean.drop(['ORA-2A-004', 'ORA-2A-036'], axis= 0)

# Drop VCCR samples
sample_mean = sample_mean.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B', 'ORA-5B-404A-B', 'ORA-5B-408-SITE8', 'ORA-5B-408-SITE7', 'ORA-5B-412B-CG'], axis= 0)

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
#sample_std = sample_std.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-032', 'ORA-2A-035', 'ORA-2A-018'], axis= 0)
#sample_std = sample_std.drop(['ORA-2A-004', 'ORA-2A-036', 'ORA-2A-032', 'ORA-2A-035'], axis= 0)
sample_std = sample_std.drop(['ORA-2A-036', 'ORA-2A-032', 'ORA-2A-035'], axis= 0)

#sample_std = sample_std.drop(['ORA-2A-004', 'ORA-2A-036'], axis= 0)

# Drop VCCR samples
sample_std = sample_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B','ORA-5B-404A-B',  'ORA-5B-408-SITE8', 'ORA-5B-408-SITE7', 'ORA-5B-412B-CG'], axis= 0)


# Select sample stdev by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCR_std = sample_std[sample_std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FG_std = sample_std[sample_std['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = sample_std[sample_std['Population'].isin(
    ['ORA-2A-002','ORA-2A-016', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

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
x = 'Rb'
y = 'Zr/Hf'

# x = 'Ba'
# y = 'Sr'

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
plot = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s',
                       edgecolor="black", s=150, alpha=0.8, legend= 'brief', hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", marker='^',
                       edgecolor="black", s=150, legend= 'brief', alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

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

#plt.text(x=MG[MG[x]], y=MG[MG[y]], s='Sample')


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

# Legend inside of plot
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

# Populations
#plt.legend(h[1:4]+h[13:16],l[1:4]+l[13:16],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Samples + populations
#plt.legend(h[0:4]+ h[5:12]+h[13:16]+h[17:27],l[0:4]+l[5:12]+ l[13:16]+l[17:27],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# General title
plt.suptitle("High-Silica Rhyolite (MG + VCCR) Fiamme Glass", fontsize=15,
             fontweight=0, color='black', y=0.95)

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/HSR-Zr_Hf_Rb_Final.png', dpi=400)

# Set size of plot
sns.set_context("paper")

plt.figure(figsize=(18, 12), dpi=400)

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