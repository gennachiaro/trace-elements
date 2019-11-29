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
df = pd.read_csv('/Users/gennachiaro/Documents/Vanderbilt/Research/Ora Caldera/Trace Elements/TraceElements_All.csv', index_col=1)

FGCP = df.loc[['ORA-2A-003','ORA-2A-002-Type2','ORA-2A-023', 'ORA-2A-024']]
FGCP_index = FGCP.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)

#create plot
plot = sns.scatterplot(data = FGCP, x= 'Ba', y='Sr', style = FGCP_index, hue = FGCP_index, palette="Greens", edgecolor="black", s=100, alpha = 0.5)

#plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend

#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("FGCP High-Silica Rhyolite Fiamme Glass", fontsize=15, fontweight=0, y=0.96)

# set size of plot
sns.set_context("paper")