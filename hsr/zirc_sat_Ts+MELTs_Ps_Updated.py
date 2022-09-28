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

##ONLY WORKS WITH BASE PYTHON 3.8.5!!!!!

# Data Cleaning
# simulated comps
# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['-']

# Zircon Sat Temps
temps = pd.read_excel(
    '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/Ora_ZircSat.xlsx', index_col=0, na_values=na_values)

temps = temps[['Sample_Name', 'Population',
               'T °C (WH 83)', 'SiO2', 'Al2O3', 'FeO', 'MgO', 'MnO', 'Na2O', 'K2O', 'TiO2', 'CaO']]

#temps = temps [['Sample_Name', 'Population', 'T °C (WH 83)']]

# Drop rows with any NaN values
temps = temps.dropna(axis=0)

#temps = temps [['Population', 'T °C (WH 83)']]

# MELTS Pressures
pressures = pd.read_excel(
    '/Users/gennachiaro/Dropbox/Rhyolite-MELTs/Final_Ora_Alteration_Simulations+Comps.xlsx', index_col=3, na_values=na_values)

pressures = pressures.drop(['ORA-5B-405-HSR-2019.10.22', 'ORA-5B-416-HSR-2019.10.23'])

# Drop columns with any NaN values
pressures = pressures.dropna(how='all', axis=1)

# Drop rows with any NaN values
pressures = pressures.dropna(axis=0)

pressures = pressures[['Name', 'Alteration_Amount', 'Pressures (MPa)', 'Depth (km)']]

#pressures = pressures [['Pressures (MPa)']]

df = temps.join(pressures, how='left')

df = df.dropna(axis = 0)

# ----------

# Plotting

# set style for boxplot
sns.set_style("darkgrid")

# # set size of plot
# fig = plt.figure(figsize=(8,4.5))

# create color dictionary
color_dict = dict({'VCCR 1': '#CC3366',
                   'VCCR 2': '#DA68A8',
                   'VCCR 3': '#D4BBDA',
                   'CG 1': '#3870AF',
                   'CG 2': '#79ADD2',
                   'CG 3': '#ABCFE5',
                   'VCCR' : '#DA68A8',
                   'MG' : '#79ADD2', 
                   'MG 3': '#ABCFE5'})

# colors = ['#DA68A8','#D4BBDA','#D4BBDA','##D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA', '#ABCFE5','#ABCFE5','#ABCFE5','#ABCFE5']
colors = ["#ABCFE5", "#ABCFE5", "#ABCFE5", "#DA68A8", "#D4BBDA", "#D4BBDA"]

# colors = ["#FF0B04", "#437B43"]
sns.set_palette(sns.color_palette(colors))

# create violin boxplot
#g = sns.violinplot(x=All.index, y="Pressures (MPa)", data=All, color='0.8', scale = 'width', inner = 'boxplot', alpha = 0.3, saturation=0.5)

#df = df.set_index('Population')
df = df.reset_index()

# Sort values by population first, and then by amount of alteration
df = df.sort_values(by=['Population'])

# Dropping VCCR 1 test

df1 = df

MG3 = df1[df1['Population'].isin(['MG 3'])]


df = df.set_index("Population")

#df = df.set_index("Sample_Name")

#df = df.drop('ORA-2A-004')
#df = df.drop('ORA-2A-032-B')


df = df.drop('MG 3')

#df = df.set_index('Population')
df = df.reset_index()

# # Sort values by population first, and then by amount of alteration
# df = df.sort_values(by=['Population'])

# calculate weights
#x_weights = np.ones_like(df['Pressures (MPa)']) / len(df['Pressures (MPa)'])

VCCR1 = df[df['Population'].isin(['VCCR 1'])]
VCCR2 = df[df['Population'].isin(['VCCR 2'])]
VCCR3 = df[df['Population'].isin(['VCCR 3'])]
MG1 = df[df['Population'].isin(['MG 1'])]
MG2 = df[df['Population'].isin(['MG 2'])]
#MG3 = df[df['Population'].isin(['MG 3'])]
CG3 = df1.loc[df1.Population == 'CG 3']


df['Type'] = df['Population'].str[:-2]
#VCCR = df[df['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
#MG = df[df['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]


replacement_mapping_dict = {
    "MG 1": "CG 1",
    "MG 2": "CG 2",
    "MG 3": "CG 3"
}

df['Population'] = df['Population'].replace(replacement_mapping_dict, regex = True)
df1['Population'] = df1['Population'].replace(replacement_mapping_dict, regex = True)

#f = sns.kdeplot(data = VCCR1, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = "Greens", alpha = 0.4)
#f = sns.kdeplot(data = VCCR2, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = "Reds", alpha = 0.4)
#f = sns.kdeplot(data = VCCR3, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True)

#f = sns.kdeplot(data = MG1, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = 'Greens', alpha = 0.4)
#f = sns.kdeplot(data = MG2, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = "Greys", alpha = 0.4)
#f = sns.kdeplot(data = MG3, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = "Greys")

# plt.ylim(reversed(plt.ylim(90,180)))

#g = sns.jointplot(data = df, x = 'T °C (WH 83)', y = 'Pressures (MPa)', palette = color_dict, hue = 'Population', kind = 'kde', shade = True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6}, marginal_kws = {'fill': True})

# hue order so blues are on top!
#g = (sns.jointplot(data = df, x = 'T °C (WH 83)', y = 'Pressures (MPa)', palette = color_dict, hue = 'Population', hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3', 'MG 1', 'MG 2'], kind = 'kde', shade = True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6}, marginal_kws = {'fill': True})).plot_joint(sns.scatterplot,style = df['Population'], s = 30).plot_joint(sns.kdeplot, zorder = 0, alpha = 0.5)

# g = (sns.jointplot(data=df, x='T °C (WH 83)', y='Pressures (MPa)', palette=color_dict, hue='Population', kind='kde', shade=True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha': 0.7}, marginal_kws={
#      'fill': True, 'common_norm' : False})).plot_joint(sns.kdeplot, zorder=0, alpha=0.3, warn_singular=False, linewidths=1).plot_joint(sns.scatterplot, style=df['Population'], s=40).set_axis_labels('T °C (WH 83)','Pressures (MPa)')

# Group by Population:
#XLIM ADDED
# g = (sns.jointplot(data=df, x='T °C (WH 83)', y='Pressures (MPa)', palette=color_dict, hue='Population', kind='kde', shade=True, xlim = [660, 800], joint_kws={"s": 100, "edgecolor": 'black', 'alpha': 0.7}, marginal_kws={
#      'fill': True, 'common_norm' : False})).plot_joint(sns.kdeplot, zorder=0, alpha=0.3, warn_singular=False, linewidths=1).plot_joint(sns.scatterplot, style=df['Population'], s=40, markers = ('o', 's', '^', 'P', 'h')).set_axis_labels('T °C (WH 83)','Pressure (MPa)')


g = (sns.jointplot(data=df, x='T °C (WH 83)', y='Pressures (MPa)', palette=color_dict, hue='Population', kind='kde', shade=True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha': 0.7}, marginal_kws={
     'fill': True, 'common_norm' : False})).plot_joint(sns.kdeplot, zorder=0, alpha=0.3, warn_singular=False, linewidths=1).plot_joint(sns.scatterplot, style=df['Population'], s=40, markers = ('o', 's', '^', 'P', 'h')).set_axis_labels('T °C (WH 83)','Pressure (MPa)')

#plt.xlim([680, 780])

g.ax_joint.scatter(data = MG3, x = 'T °C (WH 83)', y = 'Pressures (MPa)', color = '#ABCFE5', marker = 'X', s=40, edgecolor = 'white', linewidth = 0.5)

# KDE plots for population MG 3
sns.kdeplot(data = MG3, x = 'T °C (WH 83)', color = '#ABCFE5', fill = True, common_norm = False, ax = g.ax_marg_x)
sns.kdeplot(data = MG3, y = 'Pressures (MPa)', color = '#ABCFE5', fill = True, common_norm = False, ax = g.ax_marg_y)

#plt.show()


#sns.kdeplot(data = df1, x = 'Pressures (MPa)', palette=color_dict, hue='Population', fill = True, common_norm = False, ax = g.ax_marg_y, vertical = True)

# sns.histplot(VCCR1['T °C (WH 83)'], color = '#CC3366', fill = True, ax = g.ax_marg_x)
# sns.histplot(VCCR1['Pressures (MPa)'], color = '#CC3366', fill = True, ax = g.ax_marg_y, common_norm = 'False')


#plt.show()

# g = sns.JointGrid(data=df, x='T °C (WH 83)', y='Pressures (MPa)', palette=color_dict, hue='Population')
# sns.kdeplot(x, y, palette=color_dict, hue='Population', shade = 'True', shade_lowest = False, ax = g.ax_joint)



# sns.kdeplot(CG3['T °C (WH 83)'], color = '#ABCFE5', fill = True, common_norm = 'False', ax = g.ax_marg_x)
# plt.show()

# sns.histplot(VCCR1['T °C (WH 83)'], color = '#CC3366', fill = True)
# plt.show()
#sns.distplot(MG3, color = '#ABCFE5', ax = g.ax_marg_y, vertical=True)





# Group by Type
# g = (sns.jointplot(data=df, x='T °C (WH 83)', y='Pressures (MPa)', palette=color_dict, hue='Type', kind='kde', shade=True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha': 0.7}, marginal_kws={
#      'fill': True, 'common_norm' : False})).plot_joint(sns.kdeplot, zorder=0, alpha=0.3, warn_singular=False, linewidths=1).plot_joint(sns.scatterplot, style=df['Type'], s=40).set_axis_labels('T °C (WH 83)','Pressures (MPa)')

# Set definitions for making a depth axis!
def MPa2km(x):
    return x / 100 * 3.7

def km2MPa(x):
    return x / 3.7 * 100

secax = g.ax_marg_y.secondary_yaxis('right', functions = (MPa2km, km2MPa))

secax.set_ylabel('Depth (km)')



# #Plot with common norm normalizes the histograms
# If True, scale each conditional density by the number of observations such that the total area under all densities sums to 1. Otherwise, normalize each density independently

# g = (sns.jointplot(data=df, x='T °C (WH 83)', y='Pressures (MPa)', palette=color_dict, hue='Population', kind='kde', shade=True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha': 0.7}, marginal_kws={
#      'fill': True, 'common_norm' : False})).plot_joint(sns.kdeplot, zorder=0, alpha=0.3, warn_singular=False, linewidths=1).plot_joint(sns.scatterplot, style=df['Population'], s=40).set_axis_labels('T °C (WH 83)','Pressures (MPa)')


plt.ylim(reversed(plt.ylim(0, 300)))
#plt.xlim([680, 800])


#g = (sns.jointplot(data = df, x = 'T °C (WH 83)', y = 'Pressures (MPa)', palette = color_dict, hue = 'Population', kind = 'kde', joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6}, marginal_kws = {'fill': True})).plot_joint(sns.kdeplot, zorder = 0, n_levels = 6)

# plt.ylim(reversed(plt.ylim(90,180)))

#g.plot_joint(sns.kdeplot, hue = "Population", palette = color_dict, zorder = 0, levels = 6)

# h, l = plt.get_legend_handles_labels()

# l[1:4] = ('CG 1', 'CG 2', 'CG 3')

plt.legend(loc='lower left', ncol=1)


# # SECOND Y AXIS
# #plt.ylim(reversed(plt.ylim(0,16.65)))
# plt.ylim(reversed(plt.ylim(0,11.1)))

# #format x axis labels
# plt.ylabel('Depth (km)')

#_________

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


# Configure legend
#h, l = g.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[0:6],l[0:6],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


# Legend outside of plot
#plt.legend(h[0:6],l[0:6],loc='best', ncol=1)


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


# flip y-axis and set y-axis limits
# plt.ylim(reversed(plt.ylim(80,180)))

# plt.xlim(705, 740)

# format x axis labels
#g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")


# plot points for second y axis (depth)
#g = sns.stripplot(x='Population', y="Pressures (MPa)", data=All, jitter=True, ax = ax2, alpha = 0, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3', 'FG 1', 'FG 2', 'FG 3','MG 1', 'MG 2', 'FGCP 1', 'FGCP 2'], palette = color_dict)

# flip y-axis and set y-axis limits
# plt.ylim(reversed(plt.ylim(0,16.65)))
# plt.ylim(reversed(plt.ylim(0,11.1)))

#g.text(702, 5.31, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#g.text(702, 5.31, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# format x axis labels
#plt.xlabel('T °C')

# set x-axis tick spacing
# tick_spacing = 10
# g.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

#g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")

# general title
plt.suptitle("Ora Zircon Saturation Temperatures and Rhyolite-MELTS Q2F Pressures",
             fontsize=13, fontweight=0, y=1.01)

# Save Figure

#plt.show()

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/zr_satplot_temps_kde_fill+points_crust_300MPa_xlim.png', dpi=300, bbox_inches = 'tight')

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/zr_satplot_temps_CG3_300MP_final.svg', dpi=400, bbox_inches = 'tight')
