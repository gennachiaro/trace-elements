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
REE = pd.read_csv('/Users/gennachiaro/Documents/Vanderbilt/Research/Ora Caldera/Trace Elements/All_REE_Normalized.csv',index_col=0)

#plot 1
One = REE.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002']]
One_index = One.index

#plot 2
Two = REE.loc[['ORA-2A-016_Type1','ORA-2A-016_Type2','ORA-2A-016_Type3','ORA-2A-016_Type4']]
Two_index = Two.index

#plot 3
Three = REE.loc[['ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
Three_index = Three.index

tr = pd.read_csv('/Users/gennachiaro/Documents/Vanderbilt/Research/Ora Caldera/Trace Elements/TraceElements_All.csv', index_col=1)

#plot 4
Four = tr.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002']]
Four_index = Four.index

#plot 5
Five = tr.loc[['ORA-2A-016-Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4']]
Five_index = Five.index

#plot 6
Six = tr.loc[['ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
Six_index = Six.index

#set background color
sns.set_style("darkgrid")

#create plot
#set subplot structure

#plot matrix
fig = plt.figure(figsize=(10.4,11))

title = fig.suptitle("Mingled (FGCP) Fiamme Glass", fontsize=24, y = 1.025)

#plot 1
plt.subplot(3, 2, 1)
plot1 = sns.lineplot(data = One, x= 'REE', y='Sample/Chondrite', hue = One_index, sort = False, palette="Greens_d",legend="brief")
plt.title('ORA-2A-002',fontsize = 15)
plt.ylim(0.05, 200)
plt.legend(loc='lower right')
#set y axis to log scale
plot1.set(yscale='log')
#set x-axis labels


#plot 2
plt.subplot(3, 2, 3)
plot2 = sns.lineplot(data = Two, x= 'REE', y='Sample/Chondrite', hue = Two_index, sort = False, palette="Greens_d",legend="brief")
plt.title('ORA-2A-016', fontsize = 15)
plt.ylim(0.05, 200)
plt.legend(loc='lower right')
#set y axis to log scale
plot2.set(yscale='log')

#plot 3
plt.subplot(3, 2, 5)
plot3 = sns.lineplot(data = Three, x= 'REE', y='Sample/Chondrite', hue = Three_index, sort = False, palette="Greens_d",legend="brief")
plt.title('ORA-2A-024',fontsize = 15)
plt.ylim(0.05, 200)
plt.legend(loc='lower right')
#set y axis to log scale
plot3.set(yscale='log')

#plot 4
plt.subplot(3, 2, 2)
plot4 = sns.scatterplot(data = Four, x= 'Ba', y='Sr', hue = Four_index, style = Four_index, palette="Greens_r", edgecolor="black", s=100, alpha = 0.5)
plt.title('ORA-2A-002',fontsize = 15)
plt.legend(loc='lower right')

#plot 5
plt.subplot(3, 2, 4)
plot5 = sns.scatterplot(data = Five, x= 'Ba', y='Sr', hue = Five_index, style = Five_index, palette="Greens_r", edgecolor="black", s=100, alpha = 0.5)
plt.title('ORA-2A-016',fontsize = 15)
plt.legend(loc='lower right')

#plot 6
plt.subplot(3, 2, 6)
plot6 = sns.scatterplot(data = Six, x= 'U', y='Zr', hue = Six_index, style = Six_index, palette="Greens_r", edgecolor="black", s=100, alpha = 0.5, legend="brief")
plt.title('ORA-2A-024',fontsize = 15)
plt.legend(loc='upper center')

fig.tight_layout()
plt.show()


