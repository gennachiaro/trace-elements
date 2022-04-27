#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:47:20 2021

@author: gennachiaro
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Add in REE
#import xlsx file
REE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Trace_Avgs_NormalizedREE.xlsx", sheet_name = 'FGCP_Normalized')

#na_values = ['nan']

REE = REE.dropna(axis=1, how = 'all')

#  change all negatives to NaN
num = REE._get_numeric_data()
num[num <= 0] = np.nan

# drop na rows
REE = REE.dropna(axis = 0, how = 'any')
# df = df.dropna(axis=1, how='all'


#REE = REE.dropna(axis=0, how = 'any')

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
melt = (REE.melt(id_vars=['Sample','Population', 'Spot'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
#melt = melt.set_index('Sample')

#melt = melt.set_index('Sample')
#melt = melt.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG'], axis= 0)
#melt = melt.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)
#melt = melt.reset_index()

# #Drop included column
# df = df.drop('Included', axis = 1)

# # drop blank columns
# #df = df.dropna(axis = 1, how = 'all')
# df = df.dropna(axis=1, how='all')

# Dataframe Slicing using "isin"
ORA_002_REE = melt[melt['Population'].isin(['ORA-2A-002'])]
ORA_024_REE = melt[melt['Population'].isin(['ORA-2A-024'])]

#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,4))

# ---------
#plot 1
plt.subplot(1,2,1)

plt.title("ORA-2A-002", fontsize=13.5, fontweight=0, color='black', y = 0.99)

ORA_002_REE_1 = melt[melt['Sample'].isin(['ORA-2A-002-Type1'])]
ORA_002_REE_2 = melt[melt['Sample'].isin(['ORA-2A-002-Type2'])]
ORA_002_REE_3 = melt[melt['Sample'].isin(['ORA-2A-002-Type3'])]



ORA_002_REE = ORA_002_REE.replace(regex={'ORA-2A-002-Type1': 'ORA-2A-002-Type 1', 'ORA-2A-002-Type2': 'ORA-2A-002-Type 2', 'ORA-2A-002-Type3': 'ORA-2A-002-Type 3'})

#plot = sns.lineplot(data = ORA_002_REE, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens_d",legend='brief', hue_order = ('ORA-2A-002-Type 1', 'ORA-2A-002-Type 2','ORA-2A-002-Type 3'))

plot = sns.lineplot(data = ORA_002_REE_1, x= 'variable', y='value', hue = 'Spot', sort = False, palette = "Greens", alpha = 0.5, legend=False)
plot = sns.lineplot(data = ORA_002_REE_2, x= 'variable', y='value', hue = 'Spot', sort = False, palette = "Greys", alpha = 0.5, legend=False)
plot = sns.lineplot(data = ORA_002_REE_3, x= 'variable', y='value', hue = 'Spot', sort = False, palette = "PuRd", alpha = 0.5, legend=False)

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=1, handlelength = 1, columnspacing = 0.5)

plt.xlabel('')
plt.ylabel('Sample/Chondrite')

plt.text(8.1,0.12, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

props = dict(boxstyle = 'square, pad = 0.2', facecolor = 'white', alpha = 0.5, ec = 'none')
plt.text(-0.5,100, str('a'), fontsize = 13, fontweight = 'normal', bbox = props)


#set y axis to log scale
plot.set(yscale='log')
plt.ylim( (10**-1,10**2.2) )

#plt.yscale('log')


# -----------
#plot 2
plt.subplot(1,2,2)

plt.title("ORA-2A-024", fontsize=13.5, fontweight=0, color='black', y = 0.99)


ORA_024_REE = ORA_024_REE.replace(regex={'ORA-2A-024-TYPE1': 'ORA-2A-024-Type 1','ORA-2A-024-TYPE2': 'ORA-2A-024-Type 2' ,'ORA-2A-024-TYPE3': 'ORA-2A-024-Type 3','ORA-2A-024-TYPE4': 'ORA-2A-024-Type 4'})

plot = sns.lineplot(data = ORA_024_REE, x= 'variable', hue =  'Sample',  y='value', sort = False, palette="Greens_d",legend='brief')
plot = sns.lineplot(data = ORA_024_REE, x= 'variable', hue =  'Sample',  y='value', sort = False, palette="Greens_d",legend='brief')


#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
# Legend inside of plot
plt.legend(h[1:5]+h[5:8], l[1:5]+l[5:8], loc='best', ncol=1, handlelength = 1, columnspacing = 0.5)

props = dict(boxstyle = 'square, pad = 0.2', facecolor = 'white', alpha = 0.5, ec = 'none')
plt.text(-0.5,100, str('b'), fontsize = 13, fontweight = 'normal', bbox = props)


plt.xlabel('')
plt.ylabel('Sample/Chondrite')

#plot1.text(-0.5,0.45, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#plt.set_ylim(-10, 1200)

#set y axis to log scale
#plt.set_ylim (-10, 120)

plt.yscale('log')
plt.ylim( (10**-1,10**2.2) )
#plt.ylim (-10, 120)

#sns.set_context("paper") 
#plt.tight_layout()

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/Mingled_REE.png', dpi=500)
