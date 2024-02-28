#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
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
df1 = pd.read_excel(path, sheet_name = 'Data', skiprows = [1], dtype = data, na_values = na_values)

#If Value in included row is equal to zero, drop this row
df1 = df1.loc[~((df1['Included'] == 0))]

#Drop included column
df1 = df1.drop('Included', axis = 1)

# Columns converted to np.float
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

df1['Gd/Lu'] = (df1['Gd']/df1['Lu'])

df1['La/Yb'] = (df1['La']/df1['Yb'])

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

# Change analysis date to a string
s = df1['Date']
df1['Date'] = (s.dt.strftime('%Y.%m.%d'))

# Dropping Values:
df1 = df1.set_index('Sample')

#Dropping MG samples because we remeasured these
df1 = df1.drop(['ORA-2A-036-B','ORA-2A-036','ORA-2A-032','ORA-2A-035'], axis= 0)

#Dropping VCCR samples because we remeasured these
df1 = df1.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B'], axis= 0)

#ADDED THIS RECENTLY
#Drop VCCR samples because they are the same fiamma:
df1 = df1.drop(['ORA-5B-405', 'ORA-5B-416'])

# Dropping VCCR samples because we don't have matching SEM values
df1 = df1.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7', 'ORA-5B-412B-CG'], axis= 0)

# Dropping FG samples because remeasured:
df1 = df1.drop(['ORA-5B-410','ORA-5B-412B-FG'], axis= 0)
df1 = df1.reset_index()

#---------
# Calculate means for each sample (messy)
# sample_mean = df.groupby('Sample').mean()

sample_mean = df1.groupby(
    ['Sample', 'Population', 'Date']).mean()
sample_mean = sample_mean.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df1['Sample'].value_counts() #can use .size() but that includes NaN values

sample_mean = sample_mean.set_index('Sample')
sample_mean['Count'] = count

# Calculate stdev for each sample (messy)
sample_std = df1.groupby(
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

# ------
# Add in Major Elements:
# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/All_SEM_Corrected.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

# Import all corrected SEM data
df = pd.read_excel(path, index_col=1)

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
df = df.drop(['2018.07.11' , '2018.10.16','2018.07.20','2018.07.18'], axis=0)

# keep 10.02 because its ORA-2A-002!

# Drop because remeasured (repetitive)
df = df.drop(['2021.03.30'], axis=0)

# # Dropping because not enough analyses so we remeasured
df = df.drop(['2018.07.26', '2018.08.08'], axis=0)
df = df.reset_index()

# keep 7.23 for ORA_2A-023


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
df = df.drop(['Quartz Rim','Plagioclase Melt Inclusion', 'Quartz Rim', 'Type 4 Fayalite Rim', 'Glass (LSR)', 'Glass (Quartz Rim)', 'HSR Bleb', 'HSR Boundary', 'HSR (Quartz Rim)', "Quartz Melt Embayment"], axis=0)
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
VCCRm = sample_mean[sample_mean['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MGm = sample_mean[sample_mean['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
FGm = sample_mean[sample_mean['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCPm = sample_mean[sample_mean['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

MGm = MGm.reset_index()
VCCRm = VCCRm.reset_index()
FGm = FGm.reset_index()
FGCPm = FGCPm.reset_index()

MGm = MGm.set_index('Name')
VCCRm = VCCRm.set_index('Name')
FGCPm = FGCPm.set_index('Name')
FGm = FGm.set_index('Name')

# Get error bar values
# Select Standard Samples by population
MGm_std=sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
VCCRm_std=sample_std[sample_std['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FGm_std = sample_std[sample_std['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCPm_std = sample_std[sample_std['Population'].isin(['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

MGm_std = MGm_std.reset_index()
VCCRm_std = VCCRm_std.reset_index()
FGm_std = FGm_std.reset_index()
FGCPm_std = FGCPm_std.reset_index()

MGm_std = MGm_std.set_index('Name')
VCCRm_std = VCCRm_std.set_index('Name')
FGCPm_std = FGCPm_std.set_index('Name')
FGm_std = FGm_std.set_index('Name')

# Add in REE
#import xlsx file
FGCPREE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/Supplementary_Data_Table_4_Normalized_REE.xlsx", sheet_name = 'FGCP_Normalized')
HSRREE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/Supplementary_Data_Table_4_Normalized_REE.xlsx", sheet_name = 'All_Normalized')
FGREE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/Supplementary_Data_Table_4_Normalized_REE.xlsx", sheet_name = 'FG_Normalized')

#  change all negatives to NaN
num = FGREE._get_numeric_data()
num[num <= 0] = np.nan

FGREE = FGREE.dropna(axis=1, how = 'all')

FGREE = FGREE.dropna(axis=0, how = 'any')

#  change all negatives to NaN
num = HSRREE._get_numeric_data()
num[num <= 0] = np.nan

HSRREE = HSRREE.dropna(axis=1, how = 'all')
HSRREE = HSRREE.dropna(axis=0, how = 'any')


#  change all negatives to NaN
FGCPREE.loc[FGCPREE['Eu'] > 10, 'Eu'] = 0

num = FGCPREE._get_numeric_data()
num[num <= 0] = np.nan
#num[num > 10] = np.nan

FGCPREE = FGCPREE.dropna(axis=1, how = 'all')
FGCPREE = FGCPREE.dropna(axis=0, how = 'any')

#na_values = ['nan']

#FGCPREE = FGCPREE.dropna(axis=0, how = 'any')
#HSRREE = HSRREE.dropna(axis=0, how = 'any')
#FGREE = FGREE.dropna(axis=0, how = 'all')

#REE = REE.dropna(axis=1, how = 'all')

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
FGCPmelt = (FGCPREE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
HSRmelt = (HSRREE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
FGmelt = (FGREE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
#melt = melt.set_index('Sample')

#melt = melt.set_index('Sample')
#melt = melt.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG'], axis= 0)
#melt = melt.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)
#melt = melt.reset_index()



# Dataframe Slicing using "isin"
ORA_002_REE = FGCPmelt[FGCPmelt['Population'].isin(['ORA-2A-002'])]
ORA_024_REE = FGCPmelt[FGCPmelt['Population'].isin(['ORA-2A-024'])]
Unmingled = FGCPmelt[FGCPmelt['Population'].isin(['ORA-2A-023', 'ORA-2A-003'])]

# HSR subgroups

FGREE = FGmelt[FGmelt['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]


ORA_410_REE = FGmelt[FGmelt['Population'].isin(['ORA-5B-410'])]
ORA_412_REE = FGmelt[FGmelt['Population'].isin(['ORA-5B-412'])]
ORA_414_REE = FGmelt[FGmelt['Population'].isin(['ORA-5B-414'])]


FGCP_All = df1[df1['Population'].isin(['ORA-2A-002', 'ORA-2A-003','ORA-2A-023', 'ORA-2A-024'])]
FG_All = df1[df1['Population'].isin(['ORA-5B-410', 'ORA-5B-414','ORA-5B-412'])]
FG_All = FG_All[(FG_All['Sample'] != 'ORA-5B-410') & (FG_All['Sample'] != 'ORA-5B-412B-FG')]

#set background color
sns.set_style("darkgrid")

#plot matrix
#fig = plt.figure(figsize=(10.5, 7.5))

#fig = plt.figure(figsize=(10,8))
fig = plt.figure(figsize=(11,8))

# group plot title
#title = fig.suptitle("All Ora Fiamme Glass Trace Elements", fontsize=16, y=0.925)

#title = fig.suptitle("Crystal-Poor (FG + FGCP) Fiamme Glass Compositions", fontsize=16, y=0.925)

#plot 1 


# Plotting
# Select elements to plot
x = 'Nb'
y = 'U'

# y = 'La/Yb'
# x = 'La'

# xerr1 = MGm_std[x]
# yerr1 = MGm_std[y]

# # VCCR Error Bar Values
# xerr2 = VCCRm_std[x]
# yerr2 = VCCRm_std[y]

# # FGCP Error Bar Values
# xerr3 = FGCPm_std[x]
# yerr3 = FGCPm_std[y]

# # FG Error Bar Values
# xerr4 = FGm_std[x]
# yerr4 = FGm_std[y]

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
plt.subplot(2,2,1)
#   All one symbol

plot = sns.scatterplot(data = FG_All, x= x, y=y, hue = "Population", style = "Population", palette="OrRd_r", edgecolor="black", s=200, alpha = 0.1, legend=False, markers=('s', 'X', '^'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])


# plot = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
#                        edgecolor="black", s=150, legend='brief', alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
# plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plot = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
                       s=250, legend='brief', alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
             ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

# plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# plot.grid(b=True, which='major', color='w', linewidth=1.0)
# plot.grid(b=True, which='minor', color='w', linewidth=0.5)


# plot.text(34,1.1, str('error bars $\pm$ 1$\sigma$'), fontsize = 18.5, fontweight = 'normal')

# h, l = plot.get_legend_handles_labels()
# plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26], l[1:7]+l[7:10]+l[11:14] +
#            l[23:26], loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize=11)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

# plt.xticks(range(64, 82, 2))
#plt.ylim(8, 4000)

# plt.xlim(left = 0.9, right = 500)
# plt.ylim(bottom = 0.9, top =  600)


# # create major element plot
# plt.subplot(2, 2, 1)
# plot = sns.scatterplot(data=FGCPm, x='SiO2', y='K2O', hue="Population", style="Population", palette="Greens_r", legend="brief", markers=(
#     'o', 's', 'X', 'P', 'D'), edgecolor="black", s=150, alpha=0.5, hue_order=['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
# plot = sns.scatterplot(data=FGm, x='SiO2', y='K2O', hue="Population", style="Population", palette="OrRd_r", legend="brief", markers=(
#     's', '^', 'X'), edgecolor="black", alpha=0.5, s=150, hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
# plot = sns.scatterplot(data=MGm, x='SiO2', y='K2O', hue="Population", style=MGm_index,
#                        marker='^', palette="Blues_r", edgecolor="black", s=150, alpha=0.5, legend="brief")
# plot = sns.scatterplot(data=VCCRm, x='SiO2', y='K2O', hue="Population", style=VCCRm_index, marker='s',
#                        palette="PuRd_r", edgecolor="black", s=150, legend="brief", alpha=0.5, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])

# plt.xticks(range(64, 82, 2))
# plt.ylim(0.2, 4.9)

# # set location of legend
plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

# Turn on Log Axes

# plt.xscale('log')
# plt.yscale('log')

# from matplotlib.ticker import FuncFormatter
# for axis in [plot.xaxis, plot.yaxis]:
#     formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
#     axis.set_major_formatter(formatter)

#plt.ylim( (10**-1,10**3) )


# plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26], l[1:7]+l[7:10]+l[11:14] +
#            l[23:26], loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize=11)

# l[0] = "Outflow"
# l[4] = "Outflow (FGCP)"
# l[9] = "Intracaldera"
# l[13] = "Intracaldera (FG)"

h, l = plot.get_legend_handles_labels()

# l[0] = "Outflow (FGCP)"
# l[5] = 'Intracaldera (FG)'

# l[0] = "Outflow (FGCP)"
# l[5] = 'Intracaldera (FG)'

plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 15, markerscale = 1.6)

plt.legend(h[1:5] + h[6:], l[1:5] + l[6:], loc='upper left', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 14, markerscale = 1.6, title = "O-FCP                 I-FCP", title_fontsize = 15)


# l[0] = "Outflow"
# l[1:4] = ('CG 1', 'CG 2', 'CG 3')

# l[4] = "Outflow (FGCP)"
# l[9] = "Intracaldera"
# l[13] = "Intracaldera (FG)"


#plt.legend(h[0:4]+h[5:13]+h[14:],l[0:4]+l[5:13]+l[14:], loc='lower right', bbox_to_anchor=(2, -2.366), ncol=4, fontsize=11)


#plt.legend(h[0:5]+h[5:14]+h[14:],l[0:4]+l[5:14]+l[14:], loc='lower right', bbox_to_anchor=(2, -2.366), ncol=4, fontsize=11)

#plt.legend(h [1:5] + h [6:], l[1:5] + l[6:], loc='lower right', bbox_to_anchor=(2, -2.366), ncol=2, fontsize=11)

#plot 2
plt.subplot(2,2,2)
#create trace element plot

# Select elements to plot
# # x = 'Zr'
# # y = 'Th'

x = 'Hf'
y = 'Ta'

# x = 'Gd/Lu'
# y = 'La/Lu'

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
# plot2 = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
#                        edgecolor="black", s=150, alpha=0.8, legend=False, hue_order=['MG 1', 'MG 2', 'MG 3'])
# plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# plot2 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
#                        edgecolor="black", s=150, legend=False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
# plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)
plot2 = sns.scatterplot(data = FG_All, x= x, y=y, hue = "Population", style = "Population", palette="OrRd_r", edgecolor="black", s=200, alpha = 0.1, legend=False, markers=('s', 'X', '^'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])

plot2 = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
                       s=250, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
             ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


#plt.ylim(8.5,40)

plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

#plot2.text(76,4.7, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# Configure legend
#h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
h, l = plot2.get_legend_handles_labels()
#plt.legend(h[1:5] + h[6:], l[1:5] + l[6:], loc='upper left', ncol = 1, handlelength = 1, columnspacing = 1, fontsize = 14, markerscale = 1.6)

#plt.legend(h [1:5] + h [6:], l[1:5] + l[6:], loc='upper right', ncol=1, fontsize=11, columnspacing = 0.5, handlelength = 0.5)

#plt.legend(h [1:], l[1:], loc='upper right', ncol=1, fontsize=11, columnspacing = 0.5, handlelength = 0.5)

# l[0] = "Outflow (FGCP)"
# l[5] = 'Intracaldera (FG)'

# plt.legend(h, l, loc='upper right', ncol=1, fontsize=11, columnspacing = 0.5, handlelength = 0.5)


#plot 3
plt.subplot(2,2,3)

# x = 'Lu'
# y = 'La'

x = 'Ti'
y = 'U'
# x = 'Sr'
# y = 'La/Lu'

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

plot3 = sns.scatterplot(data = FG_All, x= x, y=y, hue = "Population", style = "Population", palette="OrRd_r", edgecolor="black", s=200, alpha = 0.1, legend=False, markers=('s', 'X', '^'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])

# plot3 = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
#                        edgecolor="black", s=150, alpha=0.8, legend=False, hue_order=['MG 1', 'MG 2', 'MG 3'])
# plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# plot3 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers=('h','^','P'), style = "Population",
#                        edgecolor="black", s=150, legend=False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
# plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


plot3 = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
                       s=250, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
             ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)

#plot3.text(28,11, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# Configure legend
#h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


#plot 4
plt.subplot(2,2,4)

x = 'Th'
y = 'Hf'
# x = 'Sr'
# y = 'La/Lu'

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

plot4 = sns.scatterplot(data = FG_All, x= x, y=y, hue = "Population", style = "Population", palette="OrRd_r", edgecolor="black", s=200, alpha = 0.1, legend=False, markers=('s', 'X', '^'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])

# plot3 = sns.scatterplot(data=MG, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
#                        edgecolor="black", s=150, alpha=0.8, legend=False, hue_order=['MG 1', 'MG 2', 'MG 3'])
# plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# plot3 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers=('h','^','P'), style = "Population",
#                        edgecolor="black", s=150, legend=False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
# plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


plot4 = sns.scatterplot(data=FG, x=x, y=y, hue="Population", palette="OrRd_r", style='Population', edgecolor="black",
                       s=250, legend=False, alpha=0.8, markers=('^', 'X', 's'), hue_order=['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
plt.errorbar(x=FG[x], y=FG[y], xerr=xerr4, yerr=yerr4, ls='none',
             ecolor='orange', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)



# ADD IN EXTRA SPACE FOR LOG PLOT
#plt.subplots_adjust(hspace = 0.25, wspace = 0.25)

plt.tight_layout(pad = 0.75)
#plt.show()

# set size of plot
#sns.set_context("poster")

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/FCP_4plot_4_ZR.svg', dpi=800)

