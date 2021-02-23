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
df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/ora-major-elements.csv', index_col=0)

MGm = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035', 'ORA-2A-040']]
MGm_index = MGm.index

VCCRm = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
VCCRm_index = VCCRm.index

#fine grained 
df2 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/SEM-data/majors-fgcp-grouped.csv', index_col=0)

FGCPm = df2.loc[['ORA-2A-002','ORA-2A-003','ORA-2A-016','ORA-2A-023','ORA-2A-024']]
FGCPm_index = FGCPm.index

FGm = df2.loc [['ORA-5B-414','ORA-5B-410','ORA-5B-412A','ORA-5B-412B']]
FGm_index = FGm.index

#import csv file
tr = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_All.csv', index_col=1)

FGCP = tr.loc[['ORA-2A-002-Type1','ORA-2A-002-Type2','ORA-2A-002-Type3','ORA-2A-016-Type1', 'ORA-2A-016-Type2', 'ORA-2A-016-Type3', 'ORA-2A-016-Type4','ORA-2A-003','ORA-2A-023','ORA-2A-024-TYPE1','ORA-2A-024-TYPE2','ORA-2A-024-TYPE3','ORA-2A-024-TYPE4']]   
FGCP_index = FGCP.index

MG = tr.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

VCCR = tr.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
VCCR_index = VCCR.index

FG = tr.loc [['ORA-5B-410-FG','ORA-5B-412-FG','ORA-5B-414-FG']]
FG_index = FG.index

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,7))

#group plot title
title = fig.suptitle("All Ora Fiamme Glass", fontsize=16, y = 0.925)

#plot 1 

#create major element plot
plt.subplot(2,2,1)
plot = sns.scatterplot(data = FGCPm, x= 'SiO2', y='K2O',hue = "Population", style = "Population", palette="Greens_r",legend="brief", markers = ('o','s','X','P','D'),edgecolor="black", s=150, alpha = 0.5, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plot = sns.scatterplot(data = FGm, x= 'SiO2', y='K2O',hue = "Population", style = "Population", palette="OrRd_r",legend="brief", markers = ('s', '^', 'X'), edgecolor="black", alpha = 0.5, s=150, hue_order = ['ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'])
plot = sns.scatterplot(data = MGm, x= 'SiO2', y='K2O',hue = "Population" , style = MGm_index, marker = '^', palette="Blues_r", edgecolor="black", s=150,alpha = 0.5,legend="brief")
plot = sns.scatterplot(data = VCCRm, x= 'SiO2', y='K2O',hue = "Population", style = VCCRm_index, marker = 's',palette="PuRd_r", edgecolor="black", s=150, legend = "brief", alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

plt.xticks(range(64, 82, 2))
plt.ylim(0.2, 4.9)


#set location of legend

h,l = plot.get_legend_handles_labels()
plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26],l[1:7]+l[7:10]+l[11:14]+l[23:26],loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize = 11)

#plot 2
plt.subplot(2,2,2)
#create trace element plot
plot = sns.scatterplot(data = FGCP, x= 'Ba', y='Sr', hue = "Population", style = 'Population', palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend=False, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plot = sns.scatterplot(data = MG, x= 'Ba', y='Sr',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend=False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'Ba', y='Sr',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = FG, x= 'Ba', y='Sr', style = "Population", hue = "Population", palette="OrRd_r", edgecolor="black", s=150,alpha = 0.5,legend=False, markers=('^', 'X', 's'), hue_order = ['ORA-5B-412','ORA-5B-410', 'ORA-5B-414'])

plt.ylim(-60, 590)
plt.xlim (-40,510)

#set location of legend
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plot 3
plt.subplot(2,2,3)
plot = sns.scatterplot(data = FGCP, x= 'Ba', y='Y', hue = "Population", style = 'Population', palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend=False, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plot = sns.scatterplot(data = MG, x= 'Ba', y='Y',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend=False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'Ba', y='Y',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = FG, x= 'Ba', y='Y', style = "Population", hue = "Population", palette="OrRd_r", edgecolor="black", s=150,alpha = 0.5,legend=False, markers=('^', 'X', 's'), hue_order = ['ORA-5B-412','ORA-5B-410', 'ORA-5B-414'])

plt.ylim(10,125)
plt.xlim (-40,510)


#plot 4
plt.subplot(2,2,4)
plot = sns.scatterplot(data = FGCP, x= 'Nb', y='Rb', hue = "Population", style = 'Population', palette="Greens_r", edgecolor="black", s=150, alpha = 0.5, legend=False, hue_order = ['ORA-2A-002', 'ORA-2A-003', 'ORA-2A-016', 'ORA-2A-023', 'ORA-2A-024'])
plot = sns.scatterplot(data = MG, x= 'Nb', y='Rb',hue = "Population" , style = MG_index, palette="Blues_d",marker = 's', edgecolor="black", s=150,alpha = 0.6,legend=False, hue_order = ['MG 1', 'MG 2', 'MG 3'])
plot = sns.scatterplot(data = VCCR, x= 'Nb', y='Rb',hue = "Population", palette="PuRd_r", marker = '^', edgecolor="black", s=150, legend = False, style = VCCR_index, alpha = 0.5, hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])
plot = sns.scatterplot(data = FG, x= 'Nb', y='Rb', style = "Population", hue = "Population", palette="OrRd_r", edgecolor="black", s=150,alpha = 0.5,legend=False, markers=('^', 'X', 's'), hue_order = ['ORA-5B-412','ORA-5B-410', 'ORA-5B-414'])

plt.ylim(170,780)
plt.xlim (8.1,50.8)

# set size of plot
plt.subplots_adjust(hspace = 0.25, wspace = 0.2)
plt.tight_layout(pad= 1.0)
plt.show()

# set size of plot
#sns.set_context("poster")