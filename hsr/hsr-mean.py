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

#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
#FGCP_index = FGCP.index

MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

VCCR = df.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
VCCR_index = VCCR.index

#VCCR1 = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#VCCR1_index = VCCR1.index

#MG1 = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#MG1_index = MG1.index

#import csv file
df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', index_col=1)

test = df1.loc[['ORA-2A-001']]
test_index = test.index


#FG = df.loc [['ORA-5B-410','ORA-5B-412A-FG','ORA-5B-412B-FG','ORA-5B-414-FG']]
#FG_index = FG.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (-0.2,0.4)

# set color palette
#sns.set_palette("PuBuGn_d")

#create plot
#plot = sns.scatterplot(data = VCCR1, x= 'Ba', y='Sr',hue = "Population",  style= VCCR1_index, palette="binary", marker = 'x', edgecolor="black", s=150, legend = False, alpha = 0.4, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
#plot = sns.scatterplot(data = MG1, x= 'Ba', y='Sr',hue = "Population" , style = "Population", palette="binary",marker = 'x', edgecolor="black", s=150,alpha = 0.3, legend = "brief")

#plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.2, legend = "brief")
#plot = sns.scatterplot(data = VCCR, x= 'Ba', y='Sr',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, style = VCCR_index, alpha = 0.2, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], legend = "brief")

plot = sns.pointplot( data = MG, x= 'Ba', y='Sr',hue = MG_index , palette="Blues_d",  s=150, alpha = 0.2)

#plot means
#plot = sns.catplot(kind = "point", data = MG, x= 'Ba', y='Sr',hue =MG_index , palette="Blues_d",  s=150, alpha = 0.2)
#plot = sns.pointplot(data = VCCR, x= 'Ba', y='Sr',hue = "Population",  palette="PuRd_r",  s=150, alpha = 0.2, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], legend = "brief")

#plot = sns.catplot(kind = "scatter", data = MG, x= 'Ba', y='Sr',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.2, legend = "brief")



#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend

#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

h,l = plot.get_legend_handles_labels()
#populations
plt.legend(h[1:4]+h[13:16],l[1:4]+l[13:16],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#samples + populations
#plt.legend(h[0:4]+ h[5:12]+h[13:16]+h[17:27],l[0:4]+l[5:12]+ l[13:16]+l[17:27],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

# general title
plt.suptitle("High-Silica Rhyolite (MG + VCCR) Fiamme Glass", fontsize=15, fontweight=0, color='black')

# set size of plot
sns.set_context("paper")

plt.show()