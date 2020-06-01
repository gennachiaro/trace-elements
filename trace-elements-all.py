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

FGCP = df.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002','ORA-2A-016-Type1', 'ORA-2A-016-Type2', 'ORA-2A-016-Type3', 'ORA-2A-016-Type4','ORA-2A-003','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]   
FGCP_index = FGCP.index

MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

VCCR = df.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
VCCR_index = VCCR.index

FG = df.loc [['ORA-5B-410-FG','ORA-5B-412-FG','ORA-5B-414-FG']]
FG_index = FG.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (10,500,100)

#create plot
plot = sns.scatterplot(data = FGCP, x= 'Zr', y='Y', hue = "Population", style = 'Population', palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend='brief', hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plot = sns.scatterplot(data = MG, x= 'Zr', y='Y',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend="brief", hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'Zr', y='Y',hue = "Population", palette="PuRd_d", marker = '^', edgecolor="black", s=150, legend = "brief", style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = FG, x= 'Zr', y='Y', style = FG_index, hue = "Population", palette="OrRd_r", edgecolor="black", s=150,alpha = 0.5,legend="brief", hue_order = ['ORA-5B-412','ORA-5B-410', 'ORA-5B-414'])

#set y axis to log scale
#plot.set(yscale='log')
#plot.set(xscale='log')

#set location of legend

#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

h,l = plot.get_legend_handles_labels()
#plt.legend(h[1:6]+h[7:10]+h[19:22] + h[32:35],l[1:6]+l[7:10]+l[19:22]+l[32:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot populations and samples
plt.legend(h[1:6]+h[7:10]+ h[11:18]+h[19:22] + h[23:31]+ h[32:35],l[1:6]+l[7:10]+ l[11:18]+l[19:22]+ l[23:31]+l[32:35],loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)


# general title
plt.suptitle("All Fiamme Glass", fontsize=15, fontweight=0, color='black')

# set size of plot
sns.set_context("paper")