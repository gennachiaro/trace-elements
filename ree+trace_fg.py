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

#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
#MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#VCCR = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
FG = df.loc [['ORA-5B-410','ORA-5B-412A-FG','ORA-5B-412B-FG','ORA-5B-414-FG']]

FG_index = FG.index

#import csv file
REE = pd.read_csv('/Users/gennachiaro/Documents/Vanderbilt/Research/Ora Caldera/Trace Elements/Rare Earth Elements/All_REE_Normalized.csv', index_col=0)
FGREE = REE.loc[['ORA-5B-410','ORA-5B-412A-FG','ORA-5B-412B-FG','ORA-5B-414-FG']]


#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,3.5))

#group plot title
title = fig.suptitle("Fine-Grained (FG) Fiamme Glass", fontsize=16, y = 1.04)

#plot 1 

#create ree plot
plt.subplot(1,2,1)
plot1 = sns.lineplot(data = FGREE, x= 'REE', y='Sample/Chondrite', hue = "Population", sort = False, palette="binary_d",legend="brief",hue_order = ['FG 1', 'FG 2'])
#set location of legend
plt.legend(loc='lower right')
plt.ylabel=("Sample/Chondrite")
plt.ylim(0.05, 200)

#set y axis to log scale
plot1.set(yscale='log')

#plot 2
plt.subplot(1,2,2)
#create trace element plot
plot2 = sns.scatterplot(data = FG, x= 'Ba', y='Sr', style = FG_index, hue = "Population", palette="binary_r", edgecolor="black", s=100,alpha = 0.5,legend="brief", markers=('o', 's', '^', 'X'),hue_order = ['FG 1', 'FG 2'])

#set location of legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

fig.tight_layout()
plt.show()

# set size of plot
sns.set_context("poster")