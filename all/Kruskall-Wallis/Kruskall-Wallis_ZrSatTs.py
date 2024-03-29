#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gennachiaro
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PathCollection

#simulated comps
# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['-']

df = pd.read_excel(
    '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/Ora_ZircSat_renamed.xlsx', index_col=1, na_values=na_values)

# Slice df so that I just have the HSR values
df = df[7:32]

# Drop columns with any NaN values
df = df.dropna(how = 'all',axis = 1)

# Drop rows with any NaN values
df = df.dropna(axis = 0)

df = df.reset_index()
df = df.set_index('Sample_Name')
df = df.drop(['ORA-5B-405', 'ORA-5B-416'])
df = df.reset_index()

#set style for boxplot
sns.set_style("darkgrid")

# set size of plot
fig = plt.figure(figsize=(8,4.5))

#create color dictionary
color_dict = dict({'VCCR 1': '#CC3366', 
                    'VCCR 2': '#DA68A8', 
                    'VCCR 3': '#D4BBDA', 
                    'CCR 1': '#3870AF', 
                    'CCR 2': '#79ADD2',
                    'CCR 3': '#ABCFE5'})

#colors = ['#DA68A8','#D4BBDA','#D4BBDA','##D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA', '#ABCFE5','#ABCFE5','#ABCFE5','#ABCFE5']
colors = ["#ABCFE5","#ABCFE5","#ABCFE5","#DA68A8","#D4BBDA","#D4BBDA"]

#colors = ["#FF0B04", "#437B43"]
sns.set_palette(sns.color_palette(colors))

#create violin boxplot
#g = sns.violinplot(x=All.index, y="Pressures (MPa)", data=All, color='0.8', scale = 'width', inner = 'boxplot', alpha = 0.3, saturation=0.5)

#df = df.set_index('Population')
df = df.reset_index()

# Sort values by population first, and then by amount of alteration
df = df.sort_values(by=['Population'])


#plot matrix
#fig = plt.figure(figsize=(10,7))

#plot 1

plt.subplot(1,1,1)
#plot colored in violins 
g = sns.violinplot(x='T °C (WH 83)', y="Population", palette = colors, data=df, color='.2', scale = 'width', inner = 'boxplot', alpha = 1, saturation=0.4, orient = 'h')

for artist in g.lines:
    artist.set_zorder(10)
for artist in g.findobj(PathCollection):
    artist.set_zorder(11)

#stripplot
#g = sns.stripplot(x=All.index, y="Pressures (MPa)", data=All, edgecolor = 'gray', linewidth =  0.5, jitter=True, ax = g, alpha = 0.6, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3', 'FG 1', 'FG 2', 'FG 3','MG 1', 'MG 2', 'FGCP 1', 'FGCP 2'], palette = color_dict)

#df = df.reset_index()


#xerr = df['1 sigma']

plt.ylabel('Population', fontsize = 15)
#plt.xlabel('T °C (W&H 83)', fontsize = 15)

plt.xlabel('T °C', fontsize = 15)

# Set tick font size
for label in (g.get_xticklabels() + g.get_yticklabels()):
	label.set_fontsize(12)

# plt.xticks(fontsize=12)

#swarmplot
#g = sns.swarmplot(x='Population', y="Pressures (MPa)", data=df, edgecolor = 'gray', linewidth =  0.5, ax = g, alpha = 0.7, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3','MG 1', 'MG 2','MG 3'], palette = color_dict, size = 7)
g = sns.swarmplot(x='T °C (WH 83)', y="Population", data=df, edgecolor = 'gray', linewidth =  0.5, ax = g, alpha = 0.7, hue = "Population", hue_order = ['CCR 1', 'CCR 2', 'CCR 3','VCCR 1', 'VCCR 2','VCCR 3'], palette = color_dict, size = 9)

plt.ylabel('Population', fontsize = 15)
plt.xlabel('Temperature (°C)', fontsize = 15)

# # Select group to calculate the number of observations in each group
# group = 'Sample'
# group = 'Population'

# # Calculate number of obs per group & median to position labels
# medians = df.groupby([group])['T'].median().values
# nobs = df[[group]].value_counts(sort = False).values
# nobs = [str(x) for x in nobs.tolist()]
# nobs = ["n: " + i for i in nobs]

# # Add it to the plot
# pos = range(len(nobs))
# for tick,label in zip(pos,g.get_xticklabels()):
#     g.text(pos[tick],
#           medians[tick] + 150,
#             50,
#             nobs[tick],
#             horizontalalignment='center',
#             size='small',
#             color='black',
#             weight= 300)


#set legend
#plt.legend(loc="center left", ncol = 1)


# Legend outside of plot
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

h, l = g.get_legend_handles_labels()

# l[0:2]= ('CCR 1', 'CCR 2', 'CCR 3')

plt.legend(h, l, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, handlelength = 1, columnspacing = 0.5, markerscale = 1, fontsize = 12)


# # SECOND PLOT
# plt.subplot(1,2,2)
# #plot colored in violins 
# plot2 = sns.violinplot(x='T °C (B 13)', y="Population", palette = colors, data=df, color='.2', scale = 'width', inner = 'boxplot', alpha = 1, saturation=0.4, orient = 'h')

# for artist in plot2.lines:
#     artist.set_zorder(10)
# for artist in plot2.findobj(PathCollection):
#     artist.set_zorder(11)

# #swarmplot
# #g = sns.swarmplot(x='Population', y="Pressures (MPa)", data=df, edgecolor = 'gray', linewidth =  0.5, ax = g, alpha = 0.7, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3','MG 1', 'MG 2','MG 3'], palette = color_dict, size = 7)
# plot2 = sns.swarmplot(x='T °C (B 13)', y="Population", data=df, edgecolor = 'gray', linewidth =  0.5, ax = g, alpha = 0.7, hue = "Population", hue_order = ['MG 1', 'MG 2', 'MG 3','VCCR 1', 'VCCR 2','VCCR 3'], palette = color_dict, size = 7)

# # Legend outside of plot
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


#flip y-axis and set y-axis limits
#plt.ylim(reversed(plt.ylim(0,300)))

#format x axis labels
#g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")

#create second y axis (depth)
#ax = g.twinx()
#ax.grid(False)

#plot points for second y axis (depth)
#f = sns.stripplot(x=All.index, y="Depth (km)", data=All, jitter=True, ax = ax2, alpha = 0, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3', 'FG 1', 'FG 2', 'FG 3','MG 1', 'MG 2', 'FGCP 1', 'FGCP 2'], palette = color_dict)

#flip y-axis and set y-axis limits
#plt.ylim(reversed(plt.ylim(0,16.65)))
#plt.ylim(reversed(plt.ylim(0,11.1)))

#g.text(702, 5.31, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#format x axis labels
#plt.ylabel('Depth (km)')

#set x-axis labels
#f.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")

# general title
plt.suptitle("Ora Zircon Saturation Temperatures (W&H 83)", fontsize=15, fontweight=0, y =0.95)

# Save Figure

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/zr_satplot_KW_4.svg', dpi=800, bbox_inches = 'tight')


df = df[['Population', 'T °C (WH 83)']]


import itertools

#BOEHNKE CALCULATIONS!!
#df = df[['Population', 'T °C (B 13)']]

# CG_1 = df[df['Population'].isin(['CG 1'])]
# CG_1 = CG_1[['T °C (B 13)']]
# CG_1 = CG_1.values.tolist()
# CG_1 = list(itertools.chain(*CG_1))

# CG_2 = df[df['Population'].isin(['CG 2'])]
# CG_2 = CG_2[['T °C (B 13)']]
# CG_2 = CG_2.values.tolist()
# CG_2 = list(itertools.chain(*CG_2))

# CG_3 = df[df['Population'].isin(['CG 3'])]
# CG_3 = CG_3[['T °C (B 13)']]
# CG_3 = CG_3.values.tolist()
# CG_3 = list(itertools.chain(*CG_3))

# VCCR_1 = df[df['Population'].isin(['VCCR 1'])]
# VCCR_1 = VCCR_1[['T °C (B 13)']]
# VCCR_1 = VCCR_1.values.tolist()
# VCCR_1 = list(itertools.chain(*VCCR_1))

# VCCR_2 = df[df['Population'].isin(['VCCR 2'])]
# VCCR_2 = VCCR_2[['T °C (B 13)']]
# VCCR_2 = VCCR_2.values.tolist()
# VCCR_2 = list(itertools.chain(*VCCR_2))

# VCCR_3 = df[df['Population'].isin(['VCCR 3'])]
# VCCR_3 = VCCR_3[['T °C (B 13)']]
# VCCR_3 = VCCR_3.values.tolist()
# VCCR_3 = list(itertools.chain(*VCCR_3))

CG_1 = df[df['Population'].isin(['CG 1'])]
CG_1 = CG_1[['T °C (WH 83)']]
CG_1 = CG_1.values.tolist()
CG_1 = list(itertools.chain(*CG_1))

CG_2 = df[df['Population'].isin(['CG 2'])]
CG_2 = CG_2[['T °C (WH 83)']]
CG_2 = CG_2.values.tolist()
CG_2 = list(itertools.chain(*CG_2))

CG_3 = df[df['Population'].isin(['CG 3'])]
CG_3 = CG_3[['T °C (WH 83)']]
CG_3 = CG_3.values.tolist()
CG_3 = list(itertools.chain(*CG_3))

VCCR_1 = df[df['Population'].isin(['VCCR 1'])]
VCCR_1 = VCCR_1[['T °C (WH 83)']]
VCCR_1 = VCCR_1.values.tolist()
VCCR_1 = list(itertools.chain(*VCCR_1))

VCCR_2 = df[df['Population'].isin(['VCCR 2'])]
VCCR_2 = VCCR_2[['T °C (WH 83)']]
VCCR_2 = VCCR_2.values.tolist()
VCCR_2 = list(itertools.chain(*VCCR_2))

VCCR_3 = df[df['Population'].isin(['VCCR 3'])]
VCCR_3 = VCCR_3[['T °C (WH 83)']]
VCCR_3 = VCCR_3.values.tolist()
VCCR_3 = list(itertools.chain(*VCCR_3))


from scipy import stats
result = stats.kruskal(CG_1, CG_2, CG_3)
print(result)

result = stats.kruskal(VCCR_1, VCCR_2, VCCR_3)
print(result)

#Normality Tests
stats.shapiro(VCCR_2)

stats.shapiro(VCCR_3)

stats.shapiro(CG_1)

stats.shapiro(CG_2)


stats.bartlett(VCCR_2, VCCR_3)
stats.levene(VCCR_2, VCCR_3)

stats.bartlett(CG_1, CG_2)
stats.levene(CG_1, CG_2)



stats.ttest_ind(VCCR_2, VCCR_3)
stats.ttest_ind(CG_1, CG_2)

stats.mannwhitneyu(VCCR_2, VCCR_3)
stats.mannwhitneyu(CG_1, CG_2)


import statistics

statistics.variance(VCCR_2)
statistics.variance(VCCR_3)

statistics.variance(CG_1)
statistics.variance(CG_2)
statistics.variance(CG_3)

stats.f_oneway(CG_1, CG_2, CG_3)
stats.f_oneway(VCCR_1, VCCR_2, VCCR_3)
