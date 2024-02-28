#!/bin/env python

# Created on Wed Dec 01 14:48:56 2021

# Import plotting modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import numpy as np
 

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

#ADDED THIS RECENTLY
#Drop VCCR samples because they are the same fiamma:
df = df.drop(['ORA-5B-405', 'ORA-5B-416'])

# Dropping VCCR samples because we don't have matching SEM values
df = df.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7', 'ORA-5B-412B-CG'], axis= 0)

# Dropping FG samples because remeasured:
df = df.drop(['ORA-5B-410','ORA-5B-412B-FG'], axis= 0)
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
x = 'Ba'
y = 'Sr'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]

# Create plot

#plot matrix
#fig = plt.figure(figsize=(5,8.5))
fig = plt.figure(figsize=(8.2,4))

#title = fig.suptitle("All Ora Fiamme Glass", fontsize=19, x = 0.47, y = 1.025)

# General title

#create plot
plt.subplot(1,2,1)

#Added style to the plot!
plot = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
                       edgecolor="black", s=200, alpha=0.8, legend= 'brief', hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plot = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers=('h','^','P'), style = "Population",
                       edgecolor="black", s=200, legend= 'brief', alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]', fontsize = 17)
plt.ylabel(y + " [ppm]", fontsize = 17)

plt.yticks(fontsize=12.5)
plt.xticks(fontsize=12.5)

# Configure legend
h, l = plot.get_legend_handles_labels()

l[4] = "Intracaldera"
l[0] = "Outflow"
l[1:4] = ('CCR 1', 'CCR 2', 'CCR 3')

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5, fontsize = 12)

plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='upper left', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 12.5, markerscale = 1.4, title_fontsize = 13, title = " Outflow       Intracaldera")

# plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 12.5, markerscale = 1.4)

#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 12.5, markerscale = 1.4)
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 12.5, markerscale = 1.4, title = "Intracaldera    Outflow", title_fontsize = 13)


# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/All_SEM_Corrected.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

# Import all corrected SEM data
df = pd.read_excel(path, index_col=1)
df = df[(df['Chosen'] == 1)]

# Drop "Included" column
df = df.drop(['Chosen'], axis=1)

# Drop SiO2.1 and K20.1 columns (plotting columns)
df = df.drop(['SiO2.1', 'K2O.1'], axis=1)

# Drop blank rows
df = df.dropna(axis=0, how='all')

# Drop blank columns
df = df.dropna(axis=1, how='all')

# Drop rows with any NaN values
df = df.dropna()

# Change analysis date to a string
s = df['Analysis Date']
df['Analysis Date'] = (s.dt.strftime('%Y.%m.%d'))

# create a new column titled Name with the sample name and the date
df['Name'] = df['Sample'] + "-" + df['Type'] + "-" + df['Analysis Date']

# Dropping Values:
# Drop bad analysis dates
df = df.set_index('Analysis Date')
df = df.drop(['2018.07.11', '2018.10.02', '2018.10.16','2018.07.20','2018.07.18'], axis=0)

# Drop because remeasured (repetitive)
df = df.drop(['2021.03.30'], axis=0)

# # Dropping because not enough analyses so we remeasured
df = df.drop(['2018.07.23', '2018.07.26', '2018.08.08'], axis=0)
df = df.reset_index()

# Dropping individual samples because there weren't enough analyses so we remeasured
df = df.set_index('Name')
df = df.drop(['ORA-2A-032-HSR-2018.09.04', 'ORA-2A-018-HSR-2019.10.17', 'ORA-5B-408-SITE7-HSR-2019.10.17', 'ORA-5B-415-HSR-2019.10.17'])

#ADDED RECENTLY!!!
#Drop VCCR samples because they are the same fiamma:
df = df.drop(['ORA-5B-405-HSR-2019.10.22', 'ORA-5B-416-HSR-2019.10.23'])

df = df.reset_index()

# # Dropping to match the trace element samples
# df = df.set_index('Name')
# df = df.drop([ 'ORA-2A-018-HSR-2021.09.21'])
# df = df.reset_index()


# Drop values that are not glass
df = df.set_index('Type')
df = df.drop(['Quartz Rim','Plagioclase Melt Inclusion', 'Quartz Rim'], axis=0)
df = df.reset_index()

# Calculate means for each sample (messy)
sample_mean = df.groupby(
    ['Sample', 'Name', 'Type', 'Population', 'Analysis Date']).mean()
sample_mean = sample_mean.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df['Name'].value_counts() #can use .size() but that includes NaN values

sample_mean = sample_mean.set_index('Name')
sample_mean['Count'] = count
sample_mean = sample_mean.reset_index()

# Set indexes
sample_mean = sample_mean.set_index('Sample')
# print (merge.head())

# Calculate stdev for each sample (messy)
sample_std = df.groupby(
    ['Sample', 'Name', 'Type', 'Population', 'Analysis Date']).std()
sample_std = sample_std.reset_index()

# Add in a column that tells how many samples were calculated for the stdev
sample_std = sample_std.set_index('Name')

sample_std['Count'] = count
sample_std = sample_std.reset_index()

# Multiply dataframe by two to get 2 sigma
# sample_std = sample_std *2
# print(sample_std.head())

# Merge two dataframes (stdev and populations)
# sample_std = pd.merge(populations, sample_std, how='right',
                      # left_on="Sample", right_on=sample_std.index)
# print (sample_std.head())

# Set index
sample_std = sample_std.set_index('Sample')
# print (sample_std.head())

# Plotting
#       Slicing dataframe

# Dataframe Slicing of average values using "isin"
VCCR = sample_mean[sample_mean['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MG = sample_mean[sample_mean['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
FG = sample_mean[sample_mean['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP = sample_mean[sample_mean['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

MG = MG.reset_index()
VCCR = VCCR.reset_index()
FG = FG.reset_index()
FGCP = FGCP.reset_index()

MG = MG.set_index('Name')
VCCR = VCCR.set_index('Name')
FGCP = FGCP.set_index('Name')
FG = FG.set_index('Name')

# Get error bar values
# Select Standard Samples by population
MG_std=sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCR_std=sample_std[sample_std['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FG_std = sample_std[sample_std['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = sample_std[sample_std['Population'].isin(['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

MG_std = MG_std.reset_index()
VCCR_std = VCCR_std.reset_index()
FG_std = FG_std.reset_index()
FGCP_std = FGCP_std.reset_index()

MG_std = MG_std.set_index('Name')
VCCR_std = VCCR_std.set_index('Name')
FGCP_std = FGCP_std.set_index('Name')
FG_std = FG_std.set_index('Name')

# Set up Plot
# Set background color
sns.set_style("darkgrid")

# Set axes limits
# plt.ylim(10, 50)
# plt.xlim (-0.2,0.4)

# Set color palette
# sns.set_palette("PuBuGn_d")

#create plot
plt.subplot(1,2,2)

# Plotting
# Select elements to plot
x = 'SiO2'
y = 'CaO'

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

plot = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers=('h','^','P'), style = "Population",
                       edgecolor="black", s=200, legend= False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


plot = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
                       edgecolor="black", s=200, alpha=0.8, legend= False, hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# Automatically labelling with chemical formula!
ylabel = [x] 
ylabel = [re.sub("([0-9])", "_\\1", y) for y in ylabel]
xlabel_new = ['$\mathregular{'+y+'}$' for y in ylabel]
plt.xlabel(xlabel_new[0] + ' wt%', fontsize = 17)

ylabel = [y]
ylabel = [re.sub("([0-9])", "_\\1", y) for y in ylabel]
ylabel_new = ['$\mathregular{'+y+'}$' for y in ylabel]
plt.ylabel(ylabel_new[0] + ' wt%', fontsize = 17)

plt.yticks(fontsize=12.5)
plt.xticks(fontsize=12.5)

#plot.text(1.96, 4.1, str('error bars $\pm$ 1$\sigma$'), fontsize = 15, fontweight = 'normal')

# # plot longer line for plots without all simulations
# x = np.linspace(2.4,4,100)
# y = -0.66*x +7.2
# plt.plot(x,y, 'k')

# x = np.linspace(2,3.6,100)
# y = -0.66*x +6.5
# plt.plot(x,y, 'k')

# #plt.text(2.43, 5.25, str('1:1 Alkali Exchange'), fontsize = 11, fontweight = 'normal', rotation = -37.5)

# plt.text(2.49, 4.63, str('1:1 Alkali Exchange (slope = -0.66)'), fontsize = 15, fontweight = 'normal', rotation = -36.7)

#
# plot longer line for plots without all simulations
# x = np.linspace(4.2,5.7,100)
# y = -0.66*x +6.6
# plt.plot(x,y, 'k')

# x = np.linspace(4.2,5.5,100)
# y = -0.66*x +5.65
# plt.plot(x,y, 'k')

#plot = sns.scatterplot(data=FGCP, x=x, y=y, hue="Population", palette="Greens_r", style="Population", edgecolor="black",
#                       s=150, legend=False, alpha=0.8, hue_order=['ORA-2A-002','ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
#plt.errorbar(x=FGCP[x], y=FGCP[y], xerr=xerr3, yerr=yerr3, ls='none',
#             ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

#plot = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
#                       s=150, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
#plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
#             ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# Set y axis to log scale
# plot.set(yscale='log')
# plot.set(xscale='log')

# Set location of legend
# plt.legend(loc='upper left')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Configure legend
h, l = plot.get_legend_handles_labels()

# Legend outside of plot
# plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Intracaldera"
# l[4] = "Outflow"
# l[5:7] = ('CCR 1', 'CCR 2', 'CCR 3')

# #plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5, fontsize = 12)
# plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 12.5, markerscale = 1.4)
# #plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 12.5, markerscale = 1.4, title = "Intracaldera    Outflow", title_fontsize = 13)


# # General title
# plt.suptitle("Ora Crystal-Rich (VCCR + CCR) Fiamme", fontsize=17,
#              fontweight=0, color='black', y=1)

#plt.show()

plt.tight_layout(pad = 1)

#plt.savefig('/Users/gennachiaro/Library/CloudStorage/Dropbox/Writing/Alteration Paper 2021/Adobe_Illustrator_Files/Figure_3.svg', bbox_inches='tight', dpi=800)

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/hsr-major-trace.png',  bbox_inches='tight', dpi=800)
