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

MG = df.loc[['ORA-2A-040','ORA-2A-040-DEVIT']]
MG_index = MG.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(-1, 30)
#plt.xlim (25,100)

# create color palette
#flatui= ["#A4D3EE", "#4682B4", "#104E8B","#FFE4E1", "#FFB5C5", "#CD6090"]

# set color palette
#sns.set_palette("PuBuGn_d")

#create plot
plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr',hue = MG_index , style = MG_index, palette=['#3687C1','#fd411e'],marker = 's', edgecolor="black", s=150,alpha = 0.5,legend="brief")

#plot = sns.scatterplot(data = MG1, x= 'Ba', y='Sr',hue = MG1_index , style = "Population", palette='OrRd',markers = 'X', edgecolor="black", s=150,alpha = 0.5,legend="brief")

#set y axis to log scale
#plot.set(yscale='log')

#set location of legend

#plt.legend(loc='lower right')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("ORA-2A-040", fontsize=15, fontweight=0, color='black', y = 0.96)

# set size of plot
sns.set_context("poster")