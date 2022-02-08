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

#Data Cleaning
#simulated comps
# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['-']

# import data
df = pd.read_excel(
    '/Users/gennachiaro/Desktop/Bishop_VU-GlassPT_All-Publish-2.xlsx', index_col=0, na_values=na_values)

# Drop columns with any NaN values
df = df.dropna(how = 'all',axis = 1)

# Drop rows with any NaN values
#df = df.dropna(axis = 0)

df = df[df['P_Q2F'] < 250]

# Sort values
#df = df.sort_values(by=['P_Q2F'])

#----------

#Plotting

#set style for boxplot
sns.set_style("darkgrid")

# # set size of plot
# fig = plt.figure(figsize=(8,4.5))

#create color dictionary
color_dict = dict({'EBT': '#CC3366', 
                    'LBT-East': '#3870AF', 
                    'LBT-North': '#AFE1AF'})

#colors = ['#DA68A8','#D4BBDA','#D4BBDA','##D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA','#D4BBDA', '#ABCFE5','#ABCFE5','#ABCFE5','#ABCFE5']
colors = ["#ABCFE5","#ABCFE5","#ABCFE5","#DA68A8","#D4BBDA","#D4BBDA"]

#colors = ["#FF0B04", "#437B43"]
sns.set_palette(sns.color_palette(colors))

#create violin boxplot
#g = sns.violinplot(x=All.index, y="Pressures (MPa)", data=All, color='0.8', scale = 'width', inner = 'boxplot', alpha = 0.3, saturation=0.5)


#calculate weights

EBT = df[df['Sector'].isin(['EBT'])]
LBT_East = df[df['Sector'].isin(['LBT_East'])]
LBT_North = df[df['Sector'].isin(['LBT_North'])]

#plt.show()

#f = sns.kdeplot(data = EBT, x = 'T (W&H)', y = 'P_Q2F', shade = True, cmap = "Greens", alpha = 0.4)
#f = sns.kdeplot(data = LBT_East, x = 'T (W&H)', y = 'P_Q2F', shade = True, cmap = "Reds", alpha = 0.4)
#f = sns.kdeplot(data = LBT_North, x = 'T (W&H)', y = 'P_Q2F', shade = True)

#f = sns.kdeplot(data = MG1, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = 'Greens', alpha = 0.4)
#f = sns.kdeplot(data = MG2, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = "Greys", alpha = 0.4)
#f = sns.kdeplot(data = MG3, x = 'T °C (WH 83)', y = 'Pressures (MPa)', shade = True, cmap = "Greys")



#g = sns.jointplot(data = df, x = 'T (W&H)', y = 'P_Q2F', palette = color_dict, hue = 'Sector', kind = 'kde', shade = True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6}, marginal_kws = {'fill': True})

#g = sns.jointplot(data = df, x = 'T °C (WH 83)', y = 'Pressures (MPa)', palette = color_dict, hue = 'Population', kind = 'kde', shade = True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6}, marginal_kws = {'fill': True})

#g = (sns.jointplot(data = df, x = 'T (W&H)', y = 'P_Q2F', palette = color_dict, hue = 'Sector', kind = 'kde', shade = True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6}, marginal_kws = {'fill': True})).plot_joint(sns.scatterplot())



g = sns.jointplot(data = df, x = 'T (W&H)', y = 'P_Q2F', palette = color_dict, hue = 'Sector', kind = 'kde', shade = True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6, 'xlabel': 'T °C (WH 83)', 'ylabel':'Pressures (MPa)'}, marginal_kws = {'fill': True, 'common_norm' : False}).plot_joint(sns.kdeplot, zorder = 0, alpha = 0.3, warn_singular = False, linewidths = 1).plot_joint(sns.scatterplot, s = 25, alpha = 1 , style = df['Sector']).set_axis_labels('T °C (WH 83)','Pressures (MPa)')

# Set definitions for making a depth axis!
def MPa2km(x):
    return x / 100 * 3.7

def km2MPa(x):
    return x / 3.7 * 100

secax = g.ax_marg_y.secondary_yaxis('right', functions = (MPa2km, km2MPa))

secax.set_ylabel('Depth (km)')


#g.plot_joint(sns.scatterplot)
#g = (sns.jointplot(data = df, x = 'T (W&H)', y = 'P_Q2F', palette = color_dict, hue = 'Sector', kind = 'kde', shade = True, joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6})).plot_joint(sns.scatterplot())

#g = (sns.jointplot(data = df, x = 'T °C (WH 83)', y = 'Pressures (MPa)', palette = color_dict, hue = 'Population', kind = 'scatter', joint_kws={"s": 100, "edgecolor": 'black', 'alpha':0.6}, marginal_kws = {'weights': x_weights})).plot_joint(sns.kdeplot, zorder = 0, n_levels = 6)

#plt.show()
plt.ylim(reversed(plt.ylim(0,300)))

plt.xlabel('T °C (WH 83')
plt.ylabel('Pressures (MPa)')

#g.plot_joint(sns.kdeplot, hue = "Population", palette = color_dict, zorder = 0, levels = 6)

plt.legend(loc='lower left', ncol=1)

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


#flip y-axis and set y-axis limits
# plt.ylim(reversed(plt.ylim(80,180)))

# plt.xlim(705, 740)

#format x axis labels
#g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")


# plot points for second y axis (depth)
#g = sns.stripplot(x='Population', y="Pressures (MPa)", data=All, jitter=True, ax = ax2, alpha = 0, hue = "Population", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3', 'FG 1', 'FG 2', 'FG 3','MG 1', 'MG 2', 'FGCP 1', 'FGCP 2'], palette = color_dict)

#flip y-axis and set y-axis limits
#plt.ylim(reversed(plt.ylim(0,16.65)))
#plt.ylim(reversed(plt.ylim(0,11.1)))

#g.text(702, 5.31, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#g.text(702, 5.31, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#format x axis labels
#plt.xlabel('T °C (WH 83')
#plt.ylabel('Pressures (MPa)')


#set x-axis tick spacing
# tick_spacing = 10
# g.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

#g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")

# general title
plt.suptitle("Bishop Tuff Zircon Saturation Temperatures and Rhyolite-MELTS Q2F Pressures", fontsize=13, fontweight=0, y =1.01)

# Save Figure

#plt.show()

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/zircon saturation/BT_zr_satplot_temps_kde_fill+points.png', dpi=300, bbox_inches = 'tight')
