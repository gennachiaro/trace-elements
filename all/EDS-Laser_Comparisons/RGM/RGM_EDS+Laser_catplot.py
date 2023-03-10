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
import scipy

# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/RGM_Laser_EDS_Standards.xlsx'

#path = os.path.normcase(path) # This changes path to be compatible with windows

# Master spreadsheet with clear mineral analyses removed (excel file!)
values = pd.read_excel(path)
tidy = pd.read_excel(path, sheet_name = 'Tidy')
errors = pd.read_excel(path, sheet_name = 'Errors')


# drop blank rows
#df = df.dropna(axis = 1, how = 'all')
values = values.dropna(axis=0, how='all')


# VCCR = values[values['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
# VCCR_std = errors[errors['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]

# CG = values[values['Population'].isin(['CG 1', 'CG 2', 'CG 3'])]
# CG_std = errors[errors['Population'].isin(['CG 1', 'CG 2', 'CG 3'])]


#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,4))

# group plot title
#title = fig.suptitle("All Ora Fiamme Glass Trace Elements", fontsize=16, y=0.925)
title = fig.suptitle("Comparing EDS vs. LA-ICPMS Measurements", fontsize=16, y=0.955)

#plot 1 

# Plotting
# Select elements to plot
# x = 'Ti [ppm], EDS'
# y = 'Ti [ppm], LA-ICPMS'

# xerr2 = VCCR_std[x]
# yerr2 = VCCR_std[y]


plot1 = sns.catplot(data=tidy, x='Method', y= 'Value',col = 'Element', palette="Blues_d", hue = 'Name', height = 4, aspect = 0.6, legend = False)
plot1.set_titles("{col_name} {col_var}")
plot1.set_axis_labels("Method", "[ppm]")

# plt.errorbar(x=tidy['Method'], y=tidy['Value'], xerr= errors['Method'], yerr = errors['Value'], ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# Configure legend
#h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"

# set size of plot
#plt.tight_layout(pad = 1.0)


#plt.subplots_adjust
#fig.tight_layout(pad = 3.0)

# ADD IN EXTRA SPACE FOR LOG PLOT
#plt.subplots_adjust(hspace = 0.55)


#plt.show()

# set size of plot
#sns.set_context("poster")

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/RGM_EDS_LAICPMS.png', dpi=500)

