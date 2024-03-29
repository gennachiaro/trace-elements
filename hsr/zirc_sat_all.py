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
import matplotlib.ticker as ticker


#simulated comps
# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['-']

df = pd.read_excel(
    '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/Ora_ZircSat.xlsx', index_col=1, na_values=na_values)

# Slice df so that I just have the HSR values
df = df[7:32]

# Drop columns with any NaN values
df = df.dropna(how = 'all',axis = 1)

# Drop rows with any NaN values
df = df.dropna(axis = 0)

#melt = pd.melt(df, id_vars = 'Sample', 'Population', )


#set style for boxplot
sns.set_style("darkgrid")

# set size of plot
fig = plt.figure(figsize=(8,4.5))

#create color dictionary
color_dict = dict({'VCCR 1': '#CC3366', 
                    'VCCR 2': '#DA68A8', 
                    'VCCR 3': '#D4BBDA', 
                    'MG 1': '#3870AF', 
                    'MG 2': '#79ADD2',
                    'MG 3': '#ABCFE5'})

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
fig = plt.figure(figsize=(12,5))

#plot 1

#plt.subplot(1,1,1)
#plot colored in violins 
g = sns.violinplot(x='T °C (WH 83)', y="Population", palette = colors, data=df, color='.2', scale = 'width', inner = 'boxplot', alpha = 1, saturation=0.4, orient = 'h')

g = sns.violinplot(x='T °C (B 13)', y="Population", palette = colors, data=df, color='.2', scale = 'width', inner = 'boxplot', alpha = 1, saturation=0.4, orient = 'h')

for artist in g.lines:
    artist.set_zorder(10)
for artist in g.findobj(PathCollection):
    artist.set_zorder(11)

#stripplot
#g = sns.stripplot(x=All.index, y="Pressures (MPa)", data=All, edgecolor = 'gray', linewidth =  0.5, jitter=True, ax = g, alpha = 0.6, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3', 'FG 1', 'FG 2', 'FG 3','MG 1', 'MG 2', 'FGCP 1', 'FGCP 2'], palette = color_dict)

#df = df.reset_index()


#xerr = df['1 sigma']


#swarmplot
#g = sns.swarmplot(x='Population', y="Pressures (MPa)", data=df, edgecolor = 'gray', linewidth =  0.5, ax = g, alpha = 0.7, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3','MG 1', 'MG 2','MG 3'], palette = color_dict, size = 7)
g = sns.swarmplot(x='T °C (WH 83)', y="Population", data=df, edgecolor = 'gray', linewidth =  0.5, ax = g, alpha = 0.7, hue = "Population", hue_order = ['MG 1', 'MG 2', 'MG 3','VCCR 1', 'VCCR 2','VCCR 3'], palette = color_dict, size = 7)

g = sns.swarmplot(x='T °C (B 13)', y="Population", data=df, edgecolor = 'gray', linewidth =  0.5, ax = g, alpha = 0.7, hue = "Population", hue_order = ['MG 1', 'MG 2', 'MG 3','VCCR 1', 'VCCR 2','VCCR 3'], palette = color_dict, size = 7)

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

# Configure legend
h, l = g.get_legend_handles_labels()

# Legend outside of plot
plt.legend(h[0:6],l[0:6],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


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

#g.text(702, 5.31, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#format x axis labels
plt.xlabel('T °C')

#set x-axis tick spacing
tick_spacing = 10
g.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

#g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")

# general title
plt.suptitle("Ora Zircon Saturation Temperatures", fontsize=15, fontweight=0, y =0.94)

# Save Figure

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/zr_satplot_all.png', dpi=300, bbox_inches = 'tight')
