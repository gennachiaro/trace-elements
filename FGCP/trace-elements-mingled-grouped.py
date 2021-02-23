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

One = df.loc[['ORA-2A-024-TYPE1','ORA-2A-024-TYPE2','ORA-2A-024-TYPE3','ORA-2A-024-TYPE4']]
One_index = One.index

Two = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3']]
Two_index = Two.index

Three = df.loc[['ORA-2A-016-Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4']]
Three_index = Three.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)

#create plot
#set subplot structure

#sns.scatterplot(data = One, x= 'U', y='Zr', hue = One_index, style = One_index, palette="Greens", edgecolor="black", s=100, alpha = 0.5, legend = False)
#sns.scatterplot(data = Two, x= 'Ba', y='Sr', hue = Two_index, style = Two_index, palette="Greens", edgecolor="black", s=100, alpha = 0.5, legend = False)
#sns.scatterplot(data = FGCP, x= 'Ba', y='Sr', hue = FGCP_index, style = FGCP_index, palette="Greens", edgecolor="black", s=100, alpha = 0.5, legend='brief')

#plot matrix
fig = plt.figure(figsize=(6,8))


title = fig.suptitle("Mingled (FGCP) Fiamme Glasses", fontsize=17, y = 1.02)


#plot 1
plt.subplot(3, 1, 1)
Two = sns.scatterplot(data = Two, x= 'Zr', y='Y', hue = Two_index, style = Two_index, palette="Greens_r", edgecolor="black", s=100, alpha = 0.5)
plt.title('ORA-2A-002')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
#plt.legend(loc='upper center')

#plot 2
plt.subplot(3, 1, 2)
Three = sns.scatterplot(data = Three, x= 'Zr', y='Y', hue = Three_index, style = Three_index, palette="Greens_r", edgecolor="black", s=100, alpha = 0.5)
plt.title('ORA-2A-016')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


#plot 3
plt.subplot(3, 1, 3)
One = sns.scatterplot(data = One, x= 'Zr', y='Y', hue = One_index, style = One_index, palette="Greens_r", edgecolor="black", s=100, alpha = 0.5, legend = "brief")
plt.title('ORA-2A-024')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


fig.tight_layout()
plt.show()



#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
#plt.suptitle("ORA-2A-024 Fiamme Glass", fontsize=15, fontweight=0)

# set size of plot
#sns.set_context("paper")