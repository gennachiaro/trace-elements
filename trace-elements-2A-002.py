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

FGCP = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002']]
FGCP_index = FGCP.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)

#create plot
plot = sns.scatterplot(data = FGCP, x= 'Zr', y='Y', hue = FGCP_index, style = FGCP_index, palette="Greens", edgecolor="black", s=100, alpha = 0.5, legend='brief')

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("ORA-2A-002 Fiamme Glass", fontsize=15, fontweight=0)

# set size of plot
sns.set_context("paper")