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

#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_All.csv', index_col=1)

MG = df.loc[['ORA-2A-001', 'ORA-2A-031','ORA-2A-040']]
MG_index = MG.index

DEVIT = df.loc[['ORA-2A-001-DEVIT', 'ORA-2A-031-DEVIT', 'ORA-2A-040-DEVIT']]
DEVIT_index = DEVIT.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(-1, 30)
#plt.xlim (25,100)

# create color palette
#flatui= ["#A4D3EE", "#4682B4", "#104E8B","#FFE4E1", "#FFB5C5", "#CD6090"]

# set color palette
#sns.set_palette("PuBuGn_d")

#create plot
plot = sns.scatterplot(data = MG, x= 'Nb', y='Y',hue = MG_index, style = MG_index, palette='Blues_d',marker = 's', edgecolor="black", s=150,alpha = 0.5,legend="brief")

plot = sns.scatterplot(data = DEVIT, x= 'Nb', y='Y',hue = DEVIT_index , style = DEVIT_index, palette='Reds',marker = 's', edgecolor="black", s=150,alpha = 0.5,legend="brief")

#plot = sns.scatterplot(data = MG1, x= 'Ba', y='Sr',hue = MG1_index , style = "Population", palette='OrRd',markers = 'X', edgecolor="black", s=150,alpha = 0.5,legend="brief")

#set y axis to log scale
#plot.set(yscale='log')

#set location of legend

#plt.legend(loc='lower right')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("All MG + DEVIT Fiamme Glass", fontsize=15, fontweight=0, color='black', y = 0.96)

# set size of plot
sns.set_context("poster")