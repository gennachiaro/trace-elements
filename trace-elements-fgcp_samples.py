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

#FGCP = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016-Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]   
#FGCP = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002','ORA-2A-003','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]   
FGCP = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002','ORA-2A-016-Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]   

one = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002']]   
two = df.loc[['ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4']]   
three = df.loc[['ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]   


FGCP_index = FGCP.index

one_index = one.index
two_index = two.index
three_index = three.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)

#create plot
#plot = sns.scatterplot(data = FGCP, x= 'Zr', y='Y', style = 'Population', hue = "Population", palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend="brief")

#plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot samples
plot = sns.scatterplot(data = one, x= 'Zr', y='Y', style = one_index, hue = "Population", palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend="brief")
plot = sns.scatterplot(data = two, x= 'Zr', y='Y', style = two_index, hue = "Population", palette="Blues_r", edgecolor="black", s=150, alpha = 0.5, legend="brief")
plot = sns.scatterplot(data = three, x= 'Zr', y='Y', style = three_index, hue = "Population", palette="Greys_r", edgecolor="black", s=150, alpha = 0.5, legend="brief")

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend

#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("FGCP Fiamme Glass", fontsize=15, fontweight=0, y = 0.95)

# set size of plot
sns.set_context("paper")