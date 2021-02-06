#!/opt/anaconda3/bin/myenv
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
tr = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_All.csv', index_col=1)

#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
#MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]

#VCCR = tr.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
VCCR = tr.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]

VCCR_index = VCCR.index

MG = tr.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

DEVIT = tr.loc[['ORA-2A-001-DEVIT', 'ORA-2A-031-DEVIT', 'ORA-2A-040-DEVIT']]
DEVIT_index = DEVIT.index

#import csv file
REE = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/rare-earth-elements/All_REE_Normalized.csv', index_col=0)

MGREE = REE.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MGREE_index = MGREE.index

MG1 = REE.loc[['ORA-2A-001','ORA-2A-031','ORA-2A-040']]
MG1_index = MG1.index

DEVITREE = REE.loc[['ORA-2A-001-DEVIT', 'ORA-2A-031-DEVIT', 'ORA-2A-040-DEVIT']]
DEVITREE_index = DEVITREE.index

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,3.5))

#group plot title
title = fig.suptitle("High-Silica Rhyolite (VCCR + MG) Fiamme Glass", fontsize=16, y = 1.03)

#plot 1 

#create ree plot
plt.subplot(1,2,1)
plot1 = sns.lineplot(data = MGREE, x= 'REE', y='Sample/Chondrite', hue = "Population", sort = False, palette="Blues_d",legend="brief")
plot = sns.lineplot(data = DEVITREE, x= 'REE', y='Sample/Chondrite', hue = DEVITREE_index, sort = False, palette="Reds", alpha = 0.7, legend = 'brief')


#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
#plot just populations
plt.legend(h[1:4]+h[5:9],l[1:4]+l[5:9],loc='lower right')


plt.ylabel=("Sample/Chondrite")
plt.ylim(0.05, 200)

#set y axis to log scale
plot1.set(yscale='log')

#plot 2
plt.subplot(1,2,2)
#create trace element plot
plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr',hue = "Population", palette="Blues_d", marker = '^', edgecolor="black", s=150, style = MG_index, alpha = 0.5, hue_order = ['MG 1', 'MG 2', 'MG 3'], legend = False)
plot = sns.scatterplot(data = DEVIT, x= 'Ba', y='Sr',hue = "Population" , palette="Reds",marker = 's', edgecolor="black", s=150, alpha = 0.5,style = DEVIT_index, legend = False)

#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

fig.tight_layout()
plt.show()

# set size of plot
#sns.set_context("poster")