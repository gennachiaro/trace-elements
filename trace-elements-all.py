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

FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
FGCP_index = FGCP.index

MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

VCCR = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
VCCR_index = VCCR.index

FG = df.loc [['ORA-5B-412A-FG','ORA-5B-412B-FG']]
FG_index = FG.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)

#create plot
plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=100,alpha = 0.5,legend="brief")
plot = sns.scatterplot(data = VCCR, x= 'Ba', y='Sr',hue = "Population", palette="PuRd_d", marker = '^', edgecolor="black", s=150, legend = "brief", style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = FG, x= 'Ba', y='Sr', style = FG_index, hue = "Population", palette="binary", edgecolor="black", s=100,alpha = 0.5,legend="brief", markers=('o', 's', '^', 'X'))

#plot = sns.scatterplot(data = FG, x= 'Y', y='Nb',hue = FG_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)
#plot = sns.scatterplot(data = FGCP, x= 'Y', y='Nb',hue = FGCP_index, palette="Blues",legend="brief", marker = 's', edgecolor="black", s=150)

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend

#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)

# general title
plt.suptitle("All Fiamme Glass", fontsize=15, fontweight=0, color='black')

# set size of plot
sns.set_context("paper")