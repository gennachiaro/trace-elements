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
df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_All_TEMP.csv', index_col=1)

FG = df.loc [['ORA-5B-410','ORA-5B-412-FG','ORA-5B-414-FG']]
FG_index = FG.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)

# create color palette
#flatui= ["#A4D3EE", "#4682B4", "#104E8B","#FFE4E1", "#FFB5C5", "#CD6090"]

# set color palette
#sns.set_palette("PuBuGn_d")

#create plot
plot = sns.scatterplot(data = FG, x= 'Ba', y='Zr', hue = "Population" , style = FG_index, palette="binary_r",marker = 's', edgecolor="black", s=150,alpha = 0.7, legend = "brief", hue_order = ['FG 1', 'FG 2', 'FG 3'])

#plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr', hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=100,alpha = 0.7,legend="brief")

#set y axis to log scale
#plot.set(yscale='log')

#set location of legend

#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("Fine-Grained (FG) Fiamme Glass", fontsize=15, fontweight=0, color='black', y = 0.96)

# set size of plot
sns.set_context("poster")