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

FGCP = df.loc[['ORA-2A-003','ORA-2A-024-TYPE1','ORA-2A-024-TYPE2','ORA-2A-024-TYPE3','ORA-2A-024-TYPE4']]   
FGCP_index = FGCP.index


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
fig = plt.figure(figsize=(7,11))
plt.suptitle("All Fiamme Glass", fontsize=16, fontweight=0, color='black', x= 0.45, y = 1.01)

#create plot 1
plt.subplot(3,1,1)
plot = sns.scatterplot(data = FGCP, x= 'Ba', y='Sr', hue = "Population", style = 'Population', palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend=False)

#plot 2
plt.subplot(3,1,2)
plot = sns.scatterplot(data = FGCP, x= 'Ba', y='Y', hue = "Population", style = 'Population', palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend='brief')

h,l = plot.get_legend_handles_labels()
#plot just populations
plt.legend(h[1:6]+h[7:10]+h[19:22] + h[32:35],l[1:6]+l[7:10]+l[19:22]+l[32:35],loc='center left', bbox_to_anchor=(1.03, 0.45), ncol=1)



#plot 3
plt.subplot(3,1,3)
plot = sns.scatterplot(data = FGCP, x= 'Nb', y='Rb', hue = "Population", style = 'Population', palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend=False)


# set size of plot
plt.tight_layout(pad= 1.0)
plt.show()