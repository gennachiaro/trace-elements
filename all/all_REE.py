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
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl


##FROM BA-SR-ALL.PY
# Specify pathname

# Add in REE
#import xlsx file
FGCPREE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/Supplementary_Data_Table_4_Normalized_REE.xlsx", sheet_name = 'O-FCP_Normalized')
AllREE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/Supplementary_Data_Table_4_Normalized_REE.xlsx", sheet_name = 'All_Normalized')
FGREE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/Supplementary_Data_Table_4_Normalized_REE.xlsx", sheet_name = 'I-FCP_Normalized')

#  change all negatives to NaN
num = FGREE._get_numeric_data()
num[num <= 0] = np.nan

FGREE = FGREE.dropna(axis=1, how = 'all')

FGREE = FGREE.dropna(axis=0, how = 'any')

#  change all negatives to NaN
num = AllREE._get_numeric_data()
num[num <= 0] = np.nan

AllREE = AllREE.dropna(axis=1, how = 'all')
AllREE = AllREE.dropna(axis=0, how = 'any')


#  change all negatives to NaN
FGCPREE.loc[FGCPREE['Eu'] > 10, 'Eu'] = 0

num = FGCPREE._get_numeric_data()
num[num <= 0] = np.nan
#num[num > 10] = np.nan

FGCPREE = FGCPREE.dropna(axis=1, how = 'all')
FGCPREE = FGCPREE.dropna(axis=0, how = 'any')

#na_values = ['nan']

#FGCPREE = FGCPREE.dropna(axis=0, how = 'any')
#HSRREE = HSRREE.dropna(axis=0, how = 'any')
#FGREE = FGREE.dropna(axis=0, how = 'all')

#REE = REE.dropna(axis=1, how = 'all')

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
FGCPmelt = (FGCPREE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
Allmelt = (AllREE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
FGmelt = (FGREE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
#melt = melt.set_index('Sample')

#melt = melt.set_index('Sample')
#melt = melt.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG'], axis= 0)
#melt = melt.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)
#melt = melt.reset_index()




# Dataframe Slicing using "isin"
ORA_002_REE = FGCPmelt[FGCPmelt['Population'].isin(['ORA-2A-002'])]
ORA_024_REE = FGCPmelt[FGCPmelt['Population'].isin(['ORA-2A-024'])]
Unmingled = FGCPmelt[FGCPmelt['Population'].isin(['ORA-2A-023', 'ORA-2A-003'])]

# HSR subgroups
#MG = HSRmelt[HSRmelt['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]

CCR = Allmelt[Allmelt['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]

VCCR = Allmelt[Allmelt['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]

FG = FGmelt[FGmelt['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]


ORA_410_REE = FGmelt[FGmelt['Population'].isin(['ORA-5B-410'])]
ORA_412_REE = FGmelt[FGmelt['Population'].isin(['ORA-5B-412'])]
ORA_414_REE = FGmelt[FGmelt['Population'].isin(['ORA-5B-414'])]


#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,8))

# ---------
#plot 1
#plt.subplot(3,2,1)
plt.subplot(2,2,1)

plt.title('Outflow (O-FCP* + CCR)', fontsize = 13)

ORA_002_REE = ORA_002_REE.replace(regex={'ORA-2A-002-Type1': 'ORA-2A-002-Type 1', 'ORA-2A-002-Type2': 'ORA-2A-002-Type 2', 'ORA-2A-002-Type3': 'ORA-2A-002-Type 3'})
ORA_024_REE = ORA_024_REE.replace(regex={'ORA-2A-024-TYPE1': 'ORA-2A-024-Type 1','ORA-2A-024-TYPE2': 'ORA-2A-024-Type 2' ,'ORA-2A-024-TYPE3': 'ORA-2A-024-Type 3','ORA-2A-024-TYPE4': 'ORA-2A-024-Type 4'})

plot = sns.lineplot(data = Unmingled, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens_d",legend="brief", ci = 'sd')

#plot = sns.lineplot(data = ORA_002_REE, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greys_d",legend="brief", hue_order = ('ORA-2A-002-Type 1', 'ORA-2A-002-Type 2', 'ORA-2A-002-Type 3'))
#plot = sns.lineplot(data = ORA_024_REE, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens",legend="brief")

plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()

l[0] = 'ORA-2A-002'
#l[7] = ' '
#l[9] = 'Unmingled FGCP'

# plt.legend(h[1:4]+h[5:], l[1:4]+l[5:], loc='best', ncol=1, handlelength = 1, columnspacing = 0.5, fontsize = 8)
#first_legend = plt.legend(h[1:4]+h[5:9], l[1:4]+l[5:9], loc='center right', ncol=1, handlelength = 1, columnspacing = 0.5, fontsize = 9)
#first_legend = plt.legend(h[1:9], l[1:9], loc='lower right', ncol=1, handlelength = 0.5, columnspacing = 0.5, fontsize = 10)

#first_legend = plt.legend(h[1:3], l[1:3], loc='lower right', ncol=1, handlelength = 1, columnspacing = 0.5)


plt.legend(h[1:3], l[1:3], loc='lower right', ncol=1)

# plt.gca().add_artist(first_legend)
# #plt.legend(h[10:], l[10:], loc = 'lower left', handlelength = 0.5, columnspacing = 0.5, fontsize = 10)

# plt.legend(h[4:], l[4:], loc = 'lower right', handlelength = 0.5, columnspacing = 0.5, fontsize = 10)

props = dict(boxstyle = 'square, pad = 0.2', facecolor = 'white', alpha = 0.5, ec = 'none')
plt.text(-0.5,100, str('a'), fontsize = 13, fontweight = 'normal', bbox = props)

plt.xlabel('')
plt.ylabel('Sample/Chondrite')

#plt.text(8.1,0.12, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#set y axis to log scale
plot.set(yscale='log')
plt.ylim( (10**-1,10**2.2) )

for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

#plt.yscale('log')

# ---------
#plot 2
#plt.subplot(3,2,2)
plt.subplot(2,2,2)

plt.title('Intracaldera (I-FCP + VCCR)', fontsize = 13)

FG = FG.replace(regex={'ORA-5B-410-B': 'ORA-5B-410', 'ORA-5B-412A-FG': 'ORA-5B-412', 'ORA-5B-414-FG': 'ORA-5B-414'})


plot = sns.lineplot(data = FG, x= 'variable', y='value', hue = 'Sample', sort = False, palette="OrRd_r",legend="brief", hue_order = ('ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'), ci = 'sd')

plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)

h,l = plot.get_legend_handles_labels()
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='lower right', ncol=1)

plt.xlabel('')
plt.ylabel('Sample/Chondrite')

#plt.text(-0.6,0.12, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')
plt.text(-0.6,0.12, str('error envelopes $\pm$ 1s'), fontsize = 11, fontweight = 'normal')


props = dict(boxstyle = 'square, pad = 0.2', facecolor = 'white', alpha = 0.5, ec = 'none')
plt.text(-0.5,100, str('b'), fontsize = 13, fontweight = 'normal', bbox = props)

#set y axis to log scale
plot.set(yscale='log')
plt.ylim( (10**-1,10**2.2) )

for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

#plt.yscale('log')

# ---------
#plot 3
#plt.subplot(3,2,3)
plt.subplot(2,2,3)

plot = sns.lineplot(data = CCR, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief", ci = 'sd')

#set location of legend
plt.legend(loc='lower right')

plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)

h,l = plot.get_legend_handles_labels()
plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='lower right', ncol=1)

props = dict(boxstyle = 'square, pad = 0.2', facecolor = 'white', alpha = 0.5, ec = 'none')
plt.text(-0.5,100, str('c'), fontsize = 13, fontweight = 'normal', bbox = props)

plt.xlabel('')
plt.ylabel('Sample/Chondrite')

#plt.text(8.1,0.12, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#set y axis to log scale
plot.set(yscale='log')
plt.ylim( (10**-1,10**2.2) )

for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

#plt.yscale('log')

# -----------
#plot 4
#plt.subplot(3,2,4)
plt.subplot(2,2,4)

plot = sns.lineplot(data = VCCR, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r",legend="brief", ci = 'sd')

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()
# Legend inside of plot
plt.legend(h[1:5]+h[5:8], l[1:5]+l[5:8], loc='lower right', ncol=1)

plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)

props = dict(boxstyle = 'square, pad = 0.2', facecolor = 'white', alpha = 0.5, ec = 'none')
plt.text(-0.5,100, str('d'), fontsize = 13, fontweight = 'normal', bbox = props)


plt.xlabel('')
plt.ylabel('Sample/Chondrite')

#plot1.text(-0.5,0.45, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#plt.set_ylim(-10, 1200)

#set y axis to log scale
#plt.set_ylim (-10, 120)

plt.yscale('log')
plt.ylim( (10**-1,10**2.2) )
#plt.ylim (-10, 120)

for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

# # -----------
# #plot 5
# plt.subplot(3,2,5)

# plot = sns.lineplot(data = MG, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief")
# plot = sns.lineplot(data = Unmingled, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens_d",legend="brief")

# #set location of legend
# plt.legend(loc='lower right')

# h,l = plot.get_legend_handles_labels()
# # Legend inside of plot
# plt.legend(h[1:5]+h[5:8], l[1:5]+l[5:8], loc='best', ncol=1, handlelength = 1, columnspacing = 0.5)


# plt.xlabel('')
# plt.ylabel('Sample/Chondrite')

# #plot1.text(-0.5,0.45, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# #plt.set_ylim(-10, 1200)

# #set y axis to log scale
# #plt.set_ylim (-10, 120)

# plt.yscale('log')
# plt.ylim( (10**-1,10**2.2) )
# #plt.ylim (-10, 120)


# # -----------
# #plot 6
# plt.subplot(3,2,6)

# plot = sns.lineplot(data = VCCR, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r",legend="brief")
# plot = sns.lineplot(data = FG, x= 'variable', y='value', hue = 'Sample', sort = False, palette="OrRd_r",legend="brief", hue_order = ('ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'))

# #set location of legend
# plt.legend(loc='lower right')

# h,l = plot.get_legend_handles_labels()
# # Legend inside of plot
# plt.legend(h[1:5]+h[5:8], l[1:5]+l[5:8], loc='best', ncol=1, handlelength = 1, columnspacing = 0.5)


# plt.xlabel('')
# plt.ylabel('Sample/Chondrite')

# #plot1.text(-0.5,0.45, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# #plt.set_ylim(-10, 1200)

# #set y axis to log scale
# #plt.set_ylim (-10, 120)

# plt.yscale('log')
# plt.ylim( (10**-1,10**2.2) )
# #plt.ylim (-10, 120)

# #sns.set_context("paper") 
#plt.tight_layout()

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/All_REE2.png', dpi=700)
