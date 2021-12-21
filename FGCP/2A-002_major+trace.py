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

sample_mean = sample_mean.reset_index()

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
    ['ORA-2A-002', 'ORA-2A-016', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

FGCP1 = sample_mean[sample_mean['Sample'].isin(
    ['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3'])]


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

FGCP1_std = sample_std[sample_std['Sample'].isin(
    ['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3'])]


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
# x = 'Zr'
# y = 'Y'

x = 'Zr'
y = 'Y'

# FGCP Error Bar Values
xerr3 = FGCP1_std[x]
yerr3 = FGCP1_std[y]


#_______majors
# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/All_SEM_Corrected.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

# Import all corrected SEM data
df1 = pd.read_excel(path, index_col=1)

# Drop "Included" column
df1 = df1.drop(['Chosen'], axis=1)

# Drop SiO2.1 and K20.1 columns (plotting columns)
df1 = df1.drop(['SiO2.1', 'K2O.1'], axis=1)

# Drop blank rows
df1 = df1.dropna(axis=0, how='all')

# Drop blank columns
df1 = df1.dropna(axis=1, how='all')

# Drop rows with any NaN values
df1 = df1.dropna()

# Change analysis date to a string
s = df1['Analysis Date']
df1['Analysis Date'] = (s.dt.strftime('%Y.%m.%d'))

# create a new column titled Name with the sample name and the date
df1['Name'] = df1['Sample'] + "-" + df1['Type'] + "-" + df1['Analysis Date']

# Dropping Values:
# Drop bad analysis dates
df1 = df1.set_index('Analysis Date')
#df = df.drop(['2018.07.11', '2018.10.02', '2018.10.16','2018.07.20','2018.07.18'], axis=0)
# keeping 10.02 becasue thats when we measured 2A 002
df1 = df1.drop(['2018.07.11', '2018.10.16','2018.07.20','2018.07.18'], axis=0)

# Drop because remeasured (repetitive)
df1 = df1.drop(['2021.03.30'], axis=0)

# # Dropping because not enough analyses so we remeasured
df1 = df1.drop(['2018.07.23', '2018.07.26', '2018.08.08'], axis=0)
df1 = df1.reset_index()

# Dropping individual samples because there weren't enough analyses so we remeasured
df1 = df1.set_index('Name')
df1 = df1.drop(['ORA-2A-032-HSR-2018.09.04', 'ORA-2A-018-HSR-2019.10.17', 'ORA-5B-408-SITE7-HSR-2019.10.17', 'ORA-5B-415-HSR-2019.10.17'])
df1 = df1.reset_index()

# Dropping to match the trace element samples
df1 = df1.set_index('Name')
df1 = df1.drop(['ORA-2A-004-HSR-2019.10.17', 'ORA-2A-018-HSR-2021.09.21'])
df1 = df1.reset_index()

# Drop values that are not glass
df1 = df1.set_index('Type')
df1 = df1.drop(['Quartz Rim','Plagioclase Melt Inclusion', 'Quartz Rim'], axis=0)
df1 = df1.reset_index()

# Calculate means for each sample (messy)
sample_mean1 = df1.groupby(
    ['Sample', 'Name', 'Type', 'Population', 'Analysis Date']).mean()
sample_mean1 = sample_mean1.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df1['Name'].value_counts() #can use .size() but that includes NaN values

sample_mean1 = sample_mean1.set_index('Name')
sample_mean1['Count'] = count
sample_mean1 = sample_mean1.reset_index()

# Set indexes
sample_mean1 = sample_mean1.set_index('Sample')
# print (merge.head())

# Calculate stdev for each sample (messy)
sample_std1 = df1.groupby(
    ['Sample', 'Name', 'Type', 'Population', 'Analysis Date']).std()
sample_std1 = sample_std1.reset_index()

# Add in a column that tells how many samples were calculated for the stdev
sample_std1 = sample_std1.set_index('Name')

sample_std1['Count'] = count
sample_std1 = sample_std1.reset_index()

# Multiply dataframe by two to get 2 sigma
# sample_std = sample_std *2
# print(sample_std.head())

# Set index
sample_std1 = sample_std1.set_index('Sample')
# print (sample_std.head())

# Plotting
#       Slicing dataframe

# # Dataframe Slicing of average values using "isin"
FGCPm = sample_mean1[sample_mean1['Population'].isin(
    ['ORA-2A-002'])]

FGCPm = FGCPm.reset_index()

FGCPm = FGCPm.set_index('Name')

# # Get error bar values
# # Select Standard Samples by population
FGCPm_std = sample_std1[sample_std1['Population'].isin(['ORA-2A-002'])]

FGCPm_std = FGCPm_std.reset_index()

FGCPm_std = FGCPm_std.set_index('Name')


# Set up Plot
# Set background color
sns.set_style("darkgrid")

# Set axes limits
# plt.ylim(10, 50)
# plt.xlim (-0.2,0.4)

# Set color palette
# sns.set_palette("PuBuGn_d")

# Plotting
# Select elements to plot
x = 'SiO2'
y = 'K2O'

# FGCP Error Bar Values
xerr3m = FGCPm_std[x]
yerr3m = FGCPm_std[y]


FGCP_All = df1[df1['Sample'].isin(
    ['ORA-2A-002'])]

#plot matrix
fig = plt.figure(figsize=(10,4), dpi = 400)

#create plot 1
plt.subplot(1,2,1)

#plt.figure(figsize=(4.5, 4), dpi=400)

plot = sns.scatterplot(data = FGCP_All, x= x, y= y,hue = "Type", palette= "gray", edgecolor="black", marker = '^', s=150, style = "Type", alpha = 0.2, legend = False, markers = ('o','s','X'))

#plot.text(69.3,2.56, str('error bars' + '$\pm$' + '1' + '$\sigma$'), fontsize = 13)

#plot.text(69.27,2.53, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')
plot.text(69.18,2.51, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

plot = sns.scatterplot(data=FGCPm, x=x, y=y, hue="Type", palette="Greens_r", style="Type", edgecolor="black",
                      s=200, legend=False, alpha=0.85, markers = ('o','s','X'))
plt.errorbar(x=FGCPm[x], y=FGCPm[y], xerr=xerr3m, yerr=yerr3m, ls='none',
            ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# Set tick font size
for label in (plot.get_xticklabels() + plot.get_yticklabels()):
	label.set_fontsize(11)


# plt.text(69.5,2.58, str(K2O) + 'error bars 1$\sigma$', fontsize = 13)

# plt.text(69.5,2.58, str(K2O) +'error bars' + r'\pm' + '1' + r'\sigma$', fontsize = 13)

#plot = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
#                       s=150, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
#plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
#             ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


#   Different symbol for each population
# plot = sns.scatterplot(data = VCCR, x= 'Sr', y='Ba',hue = "Population", style = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = "brief", alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

# plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
# plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)

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
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='upper right', ncol=1)

# Populations
# plt.legend(h[1:4]+h[13:16],l[1:4]+l[13:16],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Samples + populations
# plt.legend(h[0:4]+ h[5:12]+h[13:16]+h[17:27],l[0:4]+l[5:12]+ l[13:16]+l[17:27],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# General title
# plt.suptitle("ORA-2A-002 Fiamme Glasses", fontsize=15,
#              fontweight=0, color='black', y=0.95)

# Set size of plot
#sns.set_context("paper")

plt.xlabel('SiO$_2$' + ' wt.%', fontsize = 12)
plt.ylabel('K$_2$O' + " wt.%", fontsize = 12)


#plot 2
plt.subplot(1,2,2)

# Plotting
# Select elements to plot
x = 'Zr'
y = 'Y'

# Create plot
# Show all symbols
plot = sns.scatterplot(data = all_2A_002, x= x, y=y, hue = "Sample", style = "Sample", palette="gray", edgecolor="black", s=150, alpha = 0.2, legend=False, markers = ['o','X','s'], hue_order=['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3'], label = ('Type 1', 'Type 2', 'Type 3'))


#   All one symbol
# plot = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s',
#                        edgecolor="black", s=150, alpha=0.8, legend= 'brief', hue_order=['MG 1', 'MG 2', 'MG 3'])
# plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# plot = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", marker='^',
#                        edgecolor="black", s=150, legend= 'brief', alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
# plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

FGCP1 = FGCP1.replace(regex={'ORA-2A-002-Type1': 'Type 1', 'ORA-2A-002-Type2': 'Type 2', 'ORA-2A-002-Type3': 'Type 3'})

plot = sns.scatterplot(data=FGCP1, x=x, y=y, hue="Sample", palette="Greens_r", style="Sample", edgecolor="black",
                       s=200, legend='brief', alpha=0.85,  markers = ['o','s','X'])
plt.errorbar(x=FGCP1[x], y=FGCP1[y], xerr=xerr3, yerr=yerr3, ls='none',
             ecolor='green', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# plot = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
#                        s=150, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
# plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
#              ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")


#plt.text(x=MG[MG[x]], y=MG[MG[y]], s='Sample')

#plot.text(220,10, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#plot.text(5,342, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#plot.text(100,5, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

plot.text(62,53.5, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')


# Configure legend
h, l = plot.get_legend_handles_labels()


# Legend inside of plot
plt.legend(h[2:5]+h[5:8], l[2:5]+l[5:8], loc='best', ncol=1)

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/2A 002/2A-002_2plot_mt_final.png', dpi=400)


# Set size of plot
#sns.set_context("paper") 

