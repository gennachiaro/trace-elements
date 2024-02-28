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

os.chdir('/Users/gennachiaro/Library/CloudStorage/Dropbox/Research/Ora/Ora Mineral Trace Element Data/')   

# Specify pathname
data = '/Users/gennachiaro/Library/CloudStorage/Dropbox/Research/Ora/Ora Mineral Trace Element Data/Ora_Mineral_Trace_Elements-MDL-Filtered.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

na_values = ['n.d.']

# Master spreadsheet with clear mineral analyses removed (excel file!)
df = pd.read_excel(data, na_values = na_values)
df = df[df['Included'] == 1]

df = df.sort_values(by='Type', ascending=False, na_position='first')

#df = df.drop(['Site', 'Mineral', 'Included'], axis = 1)

# Drop certain rows
# wr = wr.set_index('Sample')
# wr = wr.drop(['ORA-5B-417-2'])


#df=melt = df.melt(id_vars = ('Sample', 'Phase', 'Type', 'Site', 'Mineral'), value_vars = ('Sr', 'Ba'))

# df_melt = df.melt(id_vars = ('Sample', 'Phase', 'Type', 'Site+Mineral'), value_vars = ('Sr', 'Ba'))

# df = df_melt.groupby(['Sample', 'Site+Mineral', 'Type']).mean()

# data = df_grouped.mean()

# Dataframe Slicing of average values using "isin"
ORA_031 = df[df['Sample'].isin(['ORA-2A-031'])]
ORA_409 = df[df['Sample'].isin(['ORA-5B-409'])]

ORA_031_m = ORA_031.groupby(['Mineral', 'Site', 'Phase', 'Sample', 'Type'], as_index=False).mean()
ORA_409_m = ORA_409.groupby(['Mineral', 'Site', 'Phase', 'Sample', 'Type'], as_index=False).mean()


mineral = "Plagioclase"

ORA_031_phase = ORA_031_m[ORA_031_m['Phase'] == mineral]
ORA_031_phase = ORA_031_phase.sort_values(by='Type', ascending=False, na_position='first')

ORA_409_phase = ORA_409_m[ORA_409_m['Phase']== mineral]
ORA_409_phase = ORA_409_phase.sort_values(by='Type', ascending=False, na_position='first')

ORA_031_all = ORA_031[ORA_031['Phase'] == mineral]
ORA_031_all = ORA_031_all.sort_values(by='Type', ascending=False, na_position='first')

ORA_409_all = ORA_409[ORA_409['Phase']== mineral]
ORA_409_all = ORA_409_all.sort_values(by='Type', ascending=False, na_position='first')

#All Sites
#ORA_409_all = ORA_409[(ORA_409['Site']== 4) | (ORA_409['Site']== 9)]

#ORA_409_all = ORA_409[ORA_409['Site']== 9]

#ORA_031_all = ORA_031[(ORA_031['Site']== 19) | (ORA_031['Site']== 22) | (ORA_031['Site']== 1) | (ORA_031['Site']== 18) | (ORA_031['Site']== 6)]

#ORA_031_all = ORA_031[(ORA_031['Site']== 18) | (ORA_031['Site']== 6) | (ORA_031['Site']== 22)]

#ORA_031_all = ORA_031[(ORA_031['Site']== 18) | (ORA_031['Site']== 6) | (ORA_031['Site']== 22)]


# ORA_031_all = ORA_031[(ORA_031['Site']== 22) | (ORA_031['Site']== 6)]

3ORA_031_all = ORA_031[(ORA_031['Site']== 19) | (ORA_031['Site']== 1)]


# Stats
ORA_031_std = ORA_031.groupby(['Mineral', 'Site', 'Phase', 'Sample', 'Type'], as_index=False).std()
ORA_409_std = ORA_409.groupby(['Mineral', 'Site', 'Phase', 'Sample', 'Type'], as_index=False).std()

ORA_031_std_m = ORA_031_std[ORA_031_std['Phase'] == mineral]
ORA_409_std_m = ORA_409_std[ORA_409_std['Phase']== mineral]



# ORA_031_plag = df[df['Phase'].isin(['Plagioclase'])]
# ORA_409_plag = df[df['Phase'].isin(['Plagioclase'])]


# grouped_5B_plag = ORA_409_plag.groupby(["Site+Mineral"]).mean()
# grouped_2A_plag = ORA_031_plag.groupby(["Site+Mineral"]).mean()


fig = plt.figure(figsize=(11,4))

#fig = plt.figure(figsize=(10,8))


# group plot title
#title = fig.suptitle("All Ora Fiamme Glass Trace Elements", fontsize=16, y=0.925)
# title = fig.suptitle("Whole Fiamma + Fiamme Glass", fontsize=16, y=0.925)


# title = fig.suptitle("ORA-5B-409", fontsize=18, y=1.)

#plot 1 


# Plotting
# # Select elements to plot
# x = 'SiO2'
# y = 'K2O'

y = 'Sr'
x = 'Ba'

xerr1 = ORA_031_std_m[x]
yerr1 = ORA_031_std_m[y]

# # VCCR Error Bar Values
xerr2 = ORA_409_std_m[x]
yerr2 = ORA_409_std_m[y]


# # Create plot
plt.subplot(1,2,1)

# Create plot
#   All one symbol

# plot1 = sns.scatterplot(data=ORA_031_phase, x=x, y=y, hue="Type", style = 'Type', palette="Blues_r", marker='s',
#                        edgecolor="black", s=200, alpha=0.8, legend='brief', hue_order = ['Rim', 'Core'],  markers = ('o', 'X'))
# plt.errorbar(x=ORA_031_phase[x], y=ORA_031_phase[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)


# plot1 = sns.scatterplot(data=ORA_409_phase, x=x, y=y, hue="Type",style = 'Type', palette="PuRd_r",
#                        edgecolor="black", s=200, legend='brief', alpha=0.8, hue_order = ['Rim', 'Core'],  markers = ('o', 'X'))
# plt.errorbar(x=ORA_409_phase[x], y=ORA_409_phase[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)



# No averages

# plot1 = sns.scatterplot(data=ORA_409_all, x=x, y=y, hue="Type",style = 'Type', palette="PuRd_r",
#                        edgecolor="black", s=200, legend=False, alpha=0.8, hue_order = ['Rim', 'Core'],  markers = ('o', 'X'))


plot1 = sns.scatterplot(data=ORA_031_all, x=x, y=y, hue="Type", style = 'Type', palette="Blues_r", marker='s',
                       edgecolor="black", s=200, alpha=0.8, legend='brief', hue_order = ['Rim', 'Core'],  markers = ('o', 'X'))


# plot1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# plot1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# plot1.grid(b=True, which='major', color='w', linewidth=1.0)
# plot1.grid(b=True, which='minor', color='w', linewidth=0.5)

# plt.xscale('log')
# plt.yscale('log')

# from matplotlib.ticker import FuncFormatter
# for axis in [plot1.xaxis, plot1.yaxis]:
#     formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
#     axis.set_major_formatter(formatter)

#plt.ylim( (10**-1,10**3) )

plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + ' [ppm]', fontsize = 18.5)

plot1.set_title(mineral, fontsize = 18)

h, l = plot1.get_legend_handles_labels()

plt.legend(h[1:3] + h[4:], l[1:3] + l[4:], loc = 'best', fontsize  = 14, markerscale = 1.8)


# h, l = plot.get_legend_handles_labels()
# plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26], l[1:7]+l[7:10]+l[11:14] +
#            l[23:26], loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize=11)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)


h, l = plot1.get_legend_handles_labels()


plt.yticks(fontsize=14)
plt.xticks(fontsize=14)


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


mineral = "Sanidine"

ORA_031_phase = ORA_031_m[ORA_031_m['Phase'] == mineral]
ORA_031_phase = ORA_031_phase.sort_values(by='Type', ascending=False, na_position='first')

ORA_409_phase = ORA_409_m[ORA_409_m['Phase']== mineral]
ORA_409_phase = ORA_409_phase.sort_values(by='Type', ascending=False, na_position='first')


ORA_031_std = ORA_031.groupby(['Mineral', 'Site', 'Phase', 'Sample', 'Type'], as_index=False).std()
ORA_409_std = ORA_409.groupby(['Mineral', 'Site', 'Phase', 'Sample', 'Type'], as_index=False).std()

ORA_031_phase = ORA_031_m[ORA_031_m['Phase'] == mineral]
ORA_031_phase = ORA_031_phase.sort_values(by='Type', ascending=False, na_position='first')

ORA_409_phase = ORA_409_m[ORA_409_m['Phase']== mineral]
ORA_409_phase = ORA_409_phase.sort_values(by='Type', ascending=False, na_position='first')

ORA_031_all = ORA_031[ORA_031['Phase'] == mineral]
#ORA_031_all = ORA_031[ORA_031['Type']== "Core"]

#ORA_031_all = ORA_031[(ORA_031['Phase']== mineral) & (ORA_031['Type']== "Core")]


ORA_031_all = ORA_031_all.sort_values(by='Type', ascending=False, na_position='first')

ORA_409_all = ORA_409[ORA_409['Phase']== mineral]

#ORA_409_all = ORA_409[(ORA_409['Phase']== mineral) & (ORA_409['Type']== "Core")]

ORA_409_all = ORA_409_all.sort_values(by='Type', ascending=False, na_position='first')


#ORA_031_all = ORA_031[(ORA_031['Site']== 21) | (ORA_031['Site']== 23) | (ORA_031['Site']== 8) | (ORA_031['Site']== 17)]

#ORA_031_all = ORA_031[(ORA_031['Site']== 21) | (ORA_031['Site']== 17)]

# ORA_031_all = ORA_031[(ORA_031['Site']== 8) | (ORA_031['Site']== 23)]

#ORA_409_all = ORA_409[ORA_409['Site']== 2]

#ORA_409_all = ORA_409[(ORA_409['Site']== 2) | (ORA_409['Site']== 7)]



y = 'Sr'
x = 'Ba'

xerr1 = ORA_031_std_m[x]
yerr1 = ORA_031_std_m[y]

# # VCCR Error Bar Values
xerr2 = ORA_409_std_m[x]
yerr2 = ORA_409_std_m[y]



# # Create plot
plt.subplot(1,2,2)

# Create plot
#   All one symbol
# plot2 = sns.scatterplot(data=ORA_031_phase, x=x, y=y, hue="Type", style = 'Type', palette="Blues_r", marker='s',
#                        edgecolor="black", s=200, alpha=0.8, legend=False, hue_order = ['Rim', 'Core'], markers = ('o', 'X'))
# plt.errorbar(x=ORA_031_phase[x], y=ORA_031_phase[y], xerr=xerr1, yerr=yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)


# plot2 = sns.scatterplot(data=ORA_409_phase, x=x, y=y, hue="Type", style = 'Type', palette="PuRd_r", markers = ('o', 'X'),
#                        edgecolor="black", s=200, alpha=0.8, legend=False, hue_order = ['Rim', 'Core'])
# plt.errorbar(x=ORA_409_phase[x], y=ORA_409_phase[y], xerr=xerr2, yerr=yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


plot2 = sns.scatterplot(data=ORA_031_all, x=x, y=y, hue="Type", style = 'Type', palette="Blues_r", marker='s',
                       edgecolor="black", s=200, alpha=0.8, legend="brief", hue_order = ['Rim', 'Core'], markers = ('o', 'X'))

plot2 = sns.scatterplot(data=ORA_409_all, x=x, y=y, hue="Type", style = 'Type', palette="PuRd_r", markers = ('o', 'X'),
                       edgecolor="black", s=200, alpha=0.8, legend='brief', hue_order = ['Rim', 'Core'])



#CORE ONLY
# plot2 = sns.scatterplot(data=ORA_409_all, x=x, y=y, hue="Type", style = 'Type', palette="PuRd_r", markers = ('X'),
#                        edgecolor="black", s=200, alpha=0.8, legend='brief')

# plot2 = sns.scatterplot(data=ORA_031_all, x=x, y=y, hue="Type", style = 'Type', palette="Blues_r", marker='s',
#                        edgecolor="black", s=200, alpha=0.8, legend="brief", markers = ('X'))



# plot2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# plot2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# plot2.grid(b=True, which='major', color='w', linewidth=1.0)
# plot2.grid(b=True, which='minor', color='w', linewidth=0.5)

# plt.xscale('log')
# plt.yscale('log')


# from matplotlib.ticker import FuncFormatter
# for axis in [plot2.xaxis, plot2.yaxis]:
#     formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
#     axis.set_major_formatter(formatter)

#plt.ylim( (10**-1,10**3) )

plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + ' [ppm]', fontsize = 18.5)

plot2.set_title(mineral, fontsize = 18)

h, l = plot2.get_legend_handles_labels()

plt.legend(h[1:3] + h[4:], l[1:3] + l[4:], loc = 'best', fontsize  = 14, markerscale = 1.8)

#plt.legend(h[1:2] + h[3:], l[1:2] + l[3:], loc = 'lower right', fontsize  = 14, markerscale = 1.8)


# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1)



plt.yticks(fontsize=14)
plt.xticks(fontsize=14)


# ADD IN EXTRA SPACE FOR LOG PLOT
plt.subplots_adjust(hspace = 0.25, wspace = 0.25)

#plt.subplots_adjust(bottom=0.15)
#plt.tight_layout()

#plt.show()

# set size of plot
#sns.set_context("poster")


#plt.savefig('/Users/gennachiaro/Library/CloudStorage/Dropbox/Research/Ora/Ora Mineral Trace Element Data/2a-plag_site19.png', dpi=800, bbox_inches="tight")

