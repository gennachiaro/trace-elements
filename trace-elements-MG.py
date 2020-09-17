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

#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#VCCR = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#FG = df.loc [['ORA-5B-410','ORA-5B-412A-FG','ORA-5B-412B-FG','ORA-5B-414-FG']]

MG_index = MG.index

#set background color
sns.set_style("darkgrid")

#plt.ylim(10, 50)
#plt.xlim (25,100)

# create color palette
#flatui= ["#A4D3EE", "#4682B4", "#104E8B","#FFE4E1", "#FFB5C5", "#CD6090"]

# set color palette
#sns.set_palette("PuBuGn_d")

#create plot
plot = sns.scatterplot(data = MG, x= 'Rb', y='Sr',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=100,alpha = 0.5,legend="brief")

#set y axis to log scale
#plot.set(yscale='log')

#set location of legend

#plt.legend(loc='upper left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# general title
plt.suptitle("Medium-Grained (MG) Fiamme Glass", fontsize=15, fontweight=0, color='black', y = 0.95, x = 0.5)

# set size of plot
sns.set_context("poster")