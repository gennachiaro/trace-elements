#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:32:5 2 2019

@author: gennachiaro
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#import csv file
tr = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TEST.csv', index_col=1)

#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
#MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]

#VCCR = tr.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
VCCR = tr.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]

VCCR_index = VCCR.index

MG = tr.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

#import csv file
REE = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/rare-earth-elements/All_REE_Normalized.csv', index_col=0)
#VCCRREE = REE.loc[['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
VCCRREE = REE.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]


MGREE = REE.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]


#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,7))


#group plot title
title = fig.suptitle("High-Silica Rhyolite (VCCR + MG) Fiamme Glass Populations", fontsize=16, y = 0.96)

#plot 1 

#create ree plot
plt.subplot(2,2,1)
plot = sns.lineplot(data = MGREE, x= 'REE', y='Sample/Chondrite', hue = 'Population', sort = False, palette="Blues_d",legend="brief")
plot1 = sns.lineplot(data = VCCRREE, x= 'REE', y='Sample/Chondrite', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

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
plt.subplot(2,2,2)
#create trace element plot
plot2 = sns.scatterplot(data = VCCR, x= 'Ba', y='Sr',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], legend = False)
plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr',hue = "Population" , palette="Blues_d",marker = 's', edgecolor="black", s=150, alpha = 0.5,style = MG_index, legend = False)

#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot 3
plt.subplot(2,2,3)
plot = sns.scatterplot(data = MG, x= 'U', y='Ti',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend=False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'U', y='Ti',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

#plot 4
plt.subplot(2,2,4)
plot = sns.scatterplot(data = MG, x= 'Nb', y='Y',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend=False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'Nb', y='Y',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

# set size of plot
plt.tight_layout()
plt.show()

# set size of plot
#sns.set_context("poster")