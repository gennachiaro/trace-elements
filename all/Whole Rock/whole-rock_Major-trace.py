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

os.chdir('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/')   

# Specify pathname
# Add in Major and Trace for Glass:
path = '/Users/gennachiaro/Library/CloudStorage/Dropbox/Chiaro-Dissertation/Chapter 4 - Minerals & Bulk/Supplementary Data/Ora_Whole_Rock_Major-Trace.xlsx'

# Import major trace combined sheet
na_values = ['n.d.']

wr = pd.read_excel(path, na_values = na_values)
wr = wr.set_index('Sample')

# Dataframe Slicing of average values using "isin"
VCCR_WR = wr[wr['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
CCR_WR = wr[wr['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]

#---------
# Add in Major and Trace for Glass:
path = '/Users/gennachiaro/Library/CloudStorage/Dropbox/Chiaro-Dissertation/Chapter 4 - Minerals & Bulk/Supplementary Data/Ora_Glass_Major-Trace.xlsx'

# Import major trace combined sheet
na_values = ['n.d.']

df = pd.read_excel(path, na_values = na_values)

std = pd.read_excel(path, sheet_name= "Std", na_values = na_values)

# # Drop blank rows
# df = df.dropna(axis=0, how='all')

# # Drop blank columns
# df = df.dropna(axis=1, how='all')

# # Drop rows with any NaN values
# df = df.dropna()

# Dataframe Slicing of average values using "isin"
VCCR = df[df['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
CCR = df[df['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]
FG = df[df['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP = df[df['Population'].isin(
    ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

# #FGCP = FGCP.drop(['ORA-2A-002'], axis = 0)

# Multiply dataframe by two to get 2 sigma
#sample_std = sample_std *2
# print(sample_std.head())

# Set index
std = std.set_index('Sample')

# Select sample stdev by population
CCR_std = std[std['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]
VCCR_std = std[std['Population'].isin(
    ['VCCR 1', 'VCCR 2', 'VCCR 3'])]
FG_std = std[std['Population'].isin(
    ['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FGCP_std = std[std['Population'].isin(
    ['ORA-2A-002','ORA-2A-003', 'ORA-2A-023', 'ORA-2A-024'])]

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(7.9,3.6))

#fig = plt.figure(figsize=(10,8))


# group plot title
#title = fig.suptitle("All Ora Fiamme Glass Trace Elements", fontsize=16, y=0.925)
# title = fig.suptitle("Whole Fiamma + Fiamme Glass", fontsize=16, y=0.925)
#title = fig.suptitle("Whole Fiamma + Fiamme Glass", fontsize=16, y=0.925)

#plot 1 


plt.subplot(1,2,1)
#create trace element plot

# Select elements to plot
x = 'SiO2'
y = 'Ba'

# x = 'Gd/Lu'
# y = 'La/Lu'

xerr1 = CCR_std[x]
yerr1 = CCR_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]



# Create plot
#   All one symbol
plot2 = sns.scatterplot(data=CCR, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
                       edgecolor="black", s=150, alpha=0.2, legend=False, hue_order=['CCR 1', 'CCR 2', 'CCR 3'])
plt.errorbar(x=CCR[x], y=CCR[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.2)

plot2 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
                       edgecolor="black", s=150, legend=False, alpha=0.2, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.2)

plot2 = sns.scatterplot(data=CCR_WR, x=x, y=y, hue="Population", palette="Blues_d", markers = ('o', 's', 'X'), style = "Population",
                       edgecolor="black", s=150, alpha=0.8, legend='brief', hue_order=['CCR 1', 'CCR 2', 'CCR 3'])

plot2 = sns.scatterplot(data=VCCR_WR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','P','^'), style = "Population",
                       edgecolor="black", s=150, legend='brief', alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])

# # plot line
# plt.axhline(y = 33, color = 'k', linestyle = '-')
# plt.axhline(y = 39, color = 'k', linestyle = '-')

#plot.text(220,10, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')


plt.xlabel(x + ' [ppm]')
plt.ylabel(y + " [ppm]")

#plot2.text(50,35, str('Chondrite'), fontsize = 11, fontweight = 'normal')

# Configure legend
h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

l[0] = "Outflow"
l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)

plt.legend(h[1:4] + h[5:], l[1:4] + l[5:], loc='upper right', ncol = 2, handlelength = 1, columnspacing = 0.5, fontsize = 11, markerscale = 1.2 ,title = "  Outflow     Intracaldera", title_fontsize = 12)


#plt.ylim(8.5,40)


plt.xlabel(x, fontsize = 15)
plt.ylabel(y + " [ppm]", fontsize = 15)

plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

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

#plot 3
plt.subplot(1,2,2)

x = 'Rb'
y = 'Nb'

# x = 'Sr'
# y = 'La/Lu'

xerr1 = CCR_std[x]
yerr1 = CCR_std[y]

# VCCR Error Bar Values
xerr2 = VCCR_std[x]
yerr2 = VCCR_std[y]


plot3 = sns.scatterplot(data=CCR, x=x, y=y, hue="Population", palette="Blues_d", markers = ('o', 'X', 's'), style = "Population",
                       edgecolor="black", s=150, alpha=0.2, legend=False, hue_order=['CCR 1', 'CCR 2', 'CCR 3'])
plt.errorbar(x=CCR[x], y=CCR[y], xerr=xerr1, yerr=yerr1, ls='none', ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.2)

plot3 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers=('h','^','P'), style = "Population",
                       edgecolor="black", s=150, legend=False, alpha=0.2, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
plt.errorbar(x=VCCR[x], y=VCCR[y], xerr=xerr2, yerr=yerr2, ls='none', ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.2)

plot3 = sns.scatterplot(data=CCR_WR, x=x, y=y, hue="Population", palette="Blues_d", markers = ('o', 's', 'X'), style = "Population",
                       edgecolor="black", s=150, alpha=0.8, legend=False, hue_order=['CCR 1', 'CCR 2', 'CCR 3'])

plot3 = sns.scatterplot(data=VCCR_WR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','P','^'), style = "Population",
                       edgecolor="black", s=150, legend=False, alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])

# # plot line
# plt.axhline(y = 33, color = 'k', linestyle = '-')
# plt.axhline(y = 39, color = 'k', linestyle = '-')

# plot3.text(540,35, str('Chondrite'), fontsize = 11, fontweight = 'normal')

plt.xlabel(x + ' [ppm]', fontsize = 15)
plt.ylabel(y + " [ppm]", fontsize = 15)

plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

#plot3.text(28,11, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# # Configure legend
# h, l = plot3.get_legend_handles_labels()

# # Legend outside of plot
# #plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# # Legend inside of plot
# #plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"

# plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


# set size of plot
#plt.tight_layout(pad = 0.5)
#plt.subplots_adjust
#fig.tight_layout(pad = 3.0)

# ADD IN EXTRA SPACE FOR LOG PLOT
#plt.subplots_adjust(hspace = 0.25)

plt.tight_layout(pad = 1)


#plt.show()

# set size of plot
#sns.set_context("poster")

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/whole-rock_majors-trace_1.png', dpi=800)

