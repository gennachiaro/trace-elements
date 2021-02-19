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
df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', index_col=1)

MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

VCCR = df.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
VCCR_index = VCCR.index

DEVIT = df.loc[['ORA-2A-001-DEVIT', 'ORA-2A-031-DEVIT', 'ORA-2A-040-DEVIT']]
DEVIT_index = DEVIT.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (10,500,100)

#create plot

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend

#plot matrix
#fig = plt.figure(figsize=(6.5,10))
fig = plt.figure(figsize=(6.75,11))
plt.suptitle("High-Silica Rhyolite (VCCR + MG) Fiamme Glass", fontsize=16, fontweight=0, color='black', x= 0.45, y = 1.01)

#create plot 1
plt.subplot(3,1,1)
plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend=False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'Ba', y='Sr',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = DEVIT, x= 'Ba', y='Sr', style = DEVIT_index, hue = DEVIT_index, palette="Purples_d", edgecolor="black", s=150,alpha = 0.5,legend=False, markers=('^', 'X', 's'))

plt.ylim(-2,35)
#plt.xlim (2,55)

#plot 2
plt.subplot(3,1,2)
plot = sns.scatterplot(data = MG, x= 'U', y='Ti',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend="brief", hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'U', y='Ti',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = "brief", style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = DEVIT, x= 'U', y='Ti', style = DEVIT_index, hue = DEVIT_index, palette="Purples_d", edgecolor="black", s=150,alpha = 0.5,legend="brief", markers=('^', 'X', 's'))

h,l = plot.get_legend_handles_labels()
#plot just populations
plt.legend(h[1:4]+h[13:16]+h[26:30],l[1:4]+l[13:16]+l[26:30],loc='center left', bbox_to_anchor=(1.03, 0.45), ncol=1)

plt.ylim(30,500)
plt.xlim (2,55)

#plot 3
plt.subplot(3,1,3)
plot = sns.scatterplot(data = MG, x= 'Nb', y='Y',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend=False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'Nb', y='Y',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = DEVIT, x= 'Nb', y='Y', style = DEVIT_index, hue = DEVIT_index, palette="Purples_d", edgecolor="black", s=150,alpha = 0.5,legend=False, markers=('^', 'X', 's'))

plt.ylim(10,110)
plt.xlim (5,70)

# set size of plot
plt.tight_layout(pad= 1.0)
plt.show()