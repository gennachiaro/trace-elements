#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:32:52 2019

@author: gennachiaro
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#import csv file
df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_All.csv', index_col=1)

FGCP = df.loc[['ORA-2A-016-Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4', 'ORA-2A-016-Type5']]
#FGCP_index = FGCP.index

#no type 1 or type 5:
#FGCP = df.loc[['ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4']]
FGCP_index = FGCP.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(0, 600)
#plt.xlim (0,500)
#yay = ['#33a02c','#1f78b4','#a6cee3']


# set color palette
#sns.set_palette(yay)

#create plot
#plot = sns.scatterplot(data = FGCP, x= 'Zr', y='Y', hue = FGCP_index, style = FGCP_index, palette=yay, edgecolor="black", s=150, alpha = 0.5, legend='brief')
plot = sns.scatterplot(data = FGCP, x= 'Ba', y='Sr', hue = FGCP_index, style = FGCP_index, palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend='brief')

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("ORA-2A-016", fontsize=15, fontweight=0, y = 0.94)

# set size of plot
sns.set_context("paper")
