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

FGCP1 = df.loc[['ORA-2A-016-Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4']]
FGCP1_index = FGCP1.index

FGCP2 = df.loc[['ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
FGCP2_index = FGCP2.index

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(6.5,10))
#plt.suptitle("Mingled FGCP Fiamme", fontsize=15, fontweight=0, color='black', y = 1.015)

#create plot 1
plt.subplot(3,1,1)
plot = sns.scatterplot(data = FGCP, x= 'Zr', y='Y', hue = FGCP_index, style = FGCP_index, palette="Greens", edgecolor="black", s=150, alpha = 0.5, legend='brief', hue_order = ['ORA-2A-002', 'ORA-2A-002-Type1','ORA-2A-002-Type2'])
h,l = plot.get_legend_handles_labels()
plt.legend(h[1:6],l[1:6],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.title("ORA-2A-002")


#plot 2
plt.subplot(3,1,2)
plot = sns.scatterplot(data = FGCP1, x= 'Zr', y='Y', hue = FGCP1_index, style = FGCP1_index, palette="Greens", edgecolor="black", s=150, alpha = 0.5, legend='brief')

# general title
plt.title("ORA-2A-016")

h,l = plot.get_legend_handles_labels()
plt.legend(h[1:6],l[1:6],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


#plot 3
plt.subplot(3,1,3)
plot = sns.scatterplot(data = FGCP2, x= 'Zr', y='Y', hue = FGCP2_index, style = FGCP2_index, palette="Greens", edgecolor="black", s=150, alpha = 0.5, legend='brief')
#set location of legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
# general title
plt.title("ORA-2A-024")




h,l = plot.get_legend_handles_labels()
plt.legend(h[1:6],l[1:6],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot just populations
plt.legend(h[1:6]+h[7:10]+h[19:22] + h[32:35],l[1:6]+l[7:10]+l[19:22]+l[32:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot populations and samples
#plt.legend(h[1:6]+h[7:10]+ h[11:18]+h[19:22] + h[23:31]+ h[32:35],l[1:6]+l[7:10]+ l[11:18]+l[19:22]+ l[23:31]+l[32:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

#plot populations and samples with some headers
#plt.legend(h[1:6]+h[7:10]+ h[10:18]+h[19:22] + h[22:31]+ h[31:35],l[1:6]+l[7:10]+ l[10:18]+l[19:22]+ l[22:31]+l[31:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

#plot populations and samples only sample headers
#plt.legend(h[1:6]+h[7:10]+ h[10:18]+h[19:22] + h[22:31]+ h[32:35],l[1:6]+l[7:10]+ l[10:18]+l[19:22]+ l[22:31]+l[32:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

#plot populations and samples with all headers
#plt.legend(h[1:6]+h[6:10]+ h[10:18]+h[18:22] + h[22:31]+ h[31:35],l[1:6]+l[6:10]+ l[10:18]+l[18:22]+ l[22:31]+l[31:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

#plot populations to delete and make spaces between groups
#plt.legend(h[1:6]+h[6:10]+ h[11:18]+h[18:22] + h[23:31]+ h[31:35],l[1:6]+l[6:10]+ l[11:18]+l[18:22]+ l[23:31]+l[31:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# general title
#plt.suptitle("All Fiamme Glass", fontsize=15, fontweight=0, color='black', y = 0.95)

# set size of plot
plt.tight_layout(pad= 1.0)
plt.show()