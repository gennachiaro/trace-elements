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

FGCP = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3']]
FGCP_index = FGCP.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)
fig = plt.figure(figsize=(4.5,4))

#create plot
plot = sns.scatterplot(data = FGCP, x= 'Zr', y='Y', hue = FGCP_index, style = FGCP_index, palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend="brief", markers = ['o','s','X'])

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')


#label = ["Type 1", "Type 2", "Type 3"]

#set location of legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

h,l = plot.get_legend_handles_labels()

#plot just populations
plt.legend(h[1:6]+h[7:10]+h[19:22] + h[32:35],l[1:6]+l[7:10]+l[19:22]+l[32:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
#plt.suptitle("ORA-2A-002 Fiamme Glass", fontsize=15, fontweight=0, y = 0.95)

# set size of plot
sns.set_context("paper")