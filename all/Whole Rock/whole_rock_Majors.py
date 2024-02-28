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
import os # for pathname
import re

os.chdir('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/')   

# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/Hamilton_Whole_Rock_Data_Original.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

na_values = ['n.d.']

# Master spreadsheet with clear mineral analyses removed (excel file!)
wr = pd.read_excel(path, sheet_name = 'Majors_Normalized', na_values = na_values)

# Drop certain rows
wr = wr.set_index('Sample')
wr = wr.drop(['ORA-5B-417-2'])

# Dataframe Slicing of average values using "isin"
VCCR_WR = wr[wr['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
CCR_WR = wr[wr['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]

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

#set background color
sns.set_style("darkgrid")

#plot matrix
#fig = plt.figure(figsize=(10,7.3))

#fig = plt.figure(figsize=(11,4))
fig = plt.figure(figsize=(10,4))

#fig = plt.figure(figsize=(10,8))


# group plot title
#title = fig.suptitle("All Ora Fiamme Glass Trace Elements", fontsize=16, y=0.925)
# title = fig.suptitle("Whole Fiamma + Fiamme Glass", fontsize=16, y=0.925)


# title = fig.suptitle("Whole Fiamma + Fiamme Glass Major Elements", fontsize=24, y=0.925)

#plot 1 


# Plotting
# # Select elements to plot
# x = 'SiO2'
# y = 'K2O'

y = 'K2O'
x = 'Na2O'

# xerr1 = MGm_std[x]
# yerr1 = MGm_std[y]

# # VCCR Error Bar Values
# xerr2 = VCCRm_std[x]
# yerr2 = VCCRm_std[y]


# # Create plot
plt.subplot(1,2,1)
# #   All one symbol
# plot = sns.scatterplot(data=MGm, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
#                        edgecolor="black", s=150, alpha=0.2, legend= False, hue_order=['MG 1', 'MG 2', 'MG 3'])
# plt.errorbar(x=MGm[x], y=MGm[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# plot = sns.scatterplot(data=VCCRm, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
#                        edgecolor="black", s=150, legend= False, alpha=0.2, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
# plt.errorbar(x=VCCRm[x], y=VCCRm[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

xerr1 = MGm_std[x]
yerr1 = MGm_std[y]

# VCCR Error Bar Values
xerr2 = VCCRm_std[x]
yerr2 = VCCRm_std[y]


# Create plot
#   All one symbol
plot1 = sns.scatterplot(data=MGm, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
                       edgecolor="black", s=250, alpha=0.2, legend=False, hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MGm[x], y=MGm[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.2)

plot1 = sns.scatterplot(data=VCCRm, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
                       edgecolor="black", s=250, legend=False, alpha=0.2, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCRm[x], y=VCCRm[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.2)

# CCR_WR_1 = CCR_WR.drop(['ORA-2A-031'])

plot1 = sns.scatterplot(data=CCR_WR, x=x, y=y, hue="Population", palette="Blues_d", markers = ('o', 's', 'X'), style = "Population",
                       edgecolor="black", s=250, alpha=0.8, legend='brief', hue_order=['CCR 1', 'CCR 2', 'CCR 3'])

plot1 = sns.scatterplot(data=VCCR_WR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','P','^'), style = "Population",
                       edgecolor="black", s=250, legend='brief', alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])


#plot.text(40,0.12, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# h, l = plot.get_legend_handles_labels()
# plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26], l[1:7]+l[7:10]+l[11:14] +
#            l[23:26], loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize=11)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)

# plt.xticks(range(64, 82, 2))
# plt.ylim(0.2, 4.9)


# plt.xticks(range(64, 82, 2))
# plt.ylim(0.2, 4.9)


h, l = plot1.get_legend_handles_labels()


# plt.xlabel(x + ' wt. %', fontsize = 18.5)
# plt.ylabel(y + ' wt. %', fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

# Automatically labelling with chemical formula!
ylabel = [x] 
ylabel = [re.sub("([0-9])", "_\\1", y) for y in ylabel]
xlabel_new = ['$\mathregular{'+y+'}$' for y in ylabel]
plt.xlabel(xlabel_new[0] + ' wt%', fontsize = 17)

ylabel = [y]
ylabel = [re.sub("([0-9])", "_\\1", y) for y in ylabel]
ylabel_new = ['$\mathregular{'+y+'}$' for y in ylabel]
plt.ylabel(ylabel_new[0] + ' wt%', fontsize = 17)


# Configure legend
h, l = plot1.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

l[0] = "Outflow"
l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)

plt.legend(h[1:4] + h[5:], l[1:4] + l[5:], loc='upper right', ncol = 2, handlelength = 1, columnspacing = 0.5, fontsize = 14, markerscale = 1.6 ,title = "  Outflow     Intracaldera", title_fontsize = 15)

#plt.legend(h, l, loc='upper right', ncol = 2, handlelength = 1, columnspacing = 0.5, fontsize = 14, markerscale = 1.6)


# plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26], l[1:7]+l[7:10]+l[11:14] +
#            l[23:26], loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize=11)

# l[0] = "Outflow"
# l[4] = "Outflow (FGCP)"
# l[9] = "Intracaldera"
# l[13] = "Intracaldera (FG)"


# l[0] = "Outflow"
# l[1:4] = ('CG 1', 'CG 2', 'CG 3')

# l[4] = "Outflow (FGCP)"
# l[9] = "Intracaldera"
# l[13] = "Intracaldera (FG)"


# plt.legend(h, l, loc='lower right', bbox_to_anchor=(2, -2.366), ncol=4, fontsize=11)

#plot 2
plt.subplot(1,2,2)
#create trace element plot

# Select elements to plot
x = 'SiO2'
y = 'CaO'

# x = 'Gd/Lu'
# y = 'La/Lu'

xerr1 = MGm_std[x]
yerr1 = MGm_std[y]

# VCCR Error Bar Values
xerr2 = VCCRm_std[x]
yerr2 = VCCRm_std[y]



# Create plot
#   All one symbol
plot2 = sns.scatterplot(data=MGm, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
                       edgecolor="black", s=250, alpha=0.2, legend=False, hue_order=['MG 1', 'MG 2', 'MG 3'])
plt.errorbar(x=MGm[x], y=MGm[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.2)

plot2 = sns.scatterplot(data=VCCRm, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
                       edgecolor="black", s=250, legend=False, alpha=0.2, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCRm[x], y=VCCRm[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.2)

plot2 = sns.scatterplot(data=CCR_WR, x=x, y=y, hue="Population", palette="Blues_d", markers = ('o', 's', 'X'), style = "Population",
                       edgecolor="black", s=250, alpha=0.8, legend=False, hue_order=['CCR 1', 'CCR 2', 'CCR 3'])

plot2 = sns.scatterplot(data=VCCR_WR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','P','^'), style = "Population",
                       edgecolor="black", s=250, legend=False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])

# plt.xlabel(x + ' wt. %', fontsize = 18.5)
# plt.ylabel(y + ' wt. %', fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

# Automatically labelling with chemical formula!
ylabel = [x] 
ylabel = [re.sub("([0-9])", "_\\1", y) for y in ylabel]
xlabel_new = ['$\mathregular{'+y+'}$' for y in ylabel]
plt.xlabel(xlabel_new[0] + ' wt%', fontsize = 17)

ylabel = [y]
ylabel = [re.sub("([0-9])", "_\\1", y) for y in ylabel]
ylabel_new = ['$\mathregular{'+y+'}$' for y in ylabel]
plt.ylabel(ylabel_new[0] + ' wt%', fontsize = 17)

# # Configure legend
# h, l = plot2.get_legend_handles_labels()

# # Legend outside of plot
# #plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# # Legend inside of plot
# #plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"

# #plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)

# plt.legend(h, l, loc='best', bbox_to_anchor=(1, 0.95), ncol = 1, handlelength = 1, columnspacing = 0.5, fontsize = 15, markerscale = 1.6)

# plt.legend(h, l, loc='best', bbox_to_anchor=(1, 0.95), ncol = 1, handlelength = 1, columnspacing = 0.5, fontsize = 15, markerscale = 1.6)



#plt.ylim(8.5,40)


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


# ADD IN EXTRA SPACE FOR LOG PLOT
#plt.subplots_adjust(hspace = 0.25, wspace = 0.25)
# title = fig.suptitle("Bulk Fiamma + Fiamme Glass", fontsize=22, y=1.04)



plt.tight_layout(pad = 0.75)



#plt.show()

# set size of plot
#sns.set_context("poster")

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/geochemical-plots/whole-rock_majors_2plot_2.png', dpi=800)

