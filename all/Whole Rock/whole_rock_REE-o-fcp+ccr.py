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

#import xlsx file
REE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Trace_Avgs_NormalizedREE.xlsx")

wr_REE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/Hamilton_Whole_Rock_Data_Original.xlsx", sheet_name = 'REE_Normalized')

#  change all negatives to NaN
num = REE._get_numeric_data()
num[num <= 0] = np.nan

REE = REE.dropna(axis=1, how = 'all')

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
melt = (REE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
melt = melt.set_index('Sample')

melt_wr = (wr_REE.melt(id_vars = ['Sample', 'Population'], value_vars = ['La', 'Ce', 'Nd', 'Sm', 'Dy', 'Yb'], ignore_index = False))
#melt_wr = melt_wr.set_index('Sample')


melt_wr_LaSm = melt_wr[melt_wr['variable'].isin(['La', 'Ce', 'Nd', 'Sm'])]

melt_wr_LaSm = melt_wr_LaSm.set_index('Sample')

melt_wr_DyYb = melt_wr[melt_wr['variable'].isin(['Dy', 'Yb'])]

melt_wr_DyYb = melt_wr_DyYb.set_index('Sample')

# melt_wr_Sm = melt_wr[melt_wr['variable'].isin(['La', 'Ce'])]



#melt = melt.set_index('Sample')
#melt = melt.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG'], axis= 0)
#melt = melt.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)
#melt = melt.reset_index()

# Dataframe Slicing using "isin"
VCCRREE = melt[melt['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MGREE = melt[melt['Population'].isin(['CG 1', 'CG 2', 'CG 3'])]#VCCRREE = REE.loc[['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]

VCCRREE_wr = melt_wr[melt_wr['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
MGREE_wr = melt_wr[melt_wr['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]#VCCRREE = REE.loc[['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]

VCCR_LaSm = melt_wr_LaSm[melt_wr_LaSm['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
CCR_LaSm = melt_wr_LaSm[melt_wr_LaSm['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]

VCCR_DyYb = melt_wr_DyYb[melt_wr_DyYb['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
CCR_DyYb = melt_wr_DyYb[melt_wr_DyYb['Population'].isin(['CCR 1', 'CCR 2', 'CCR 3'])]


VCCR_Glass_LaSm = VCCRREE[VCCRREE['variable'].isin(['La', 'Ce', 'Nd', 'Sm'])]
CCR_Glass_LaSm = MGREE[MGREE['variable'].isin(['La', 'Ce', 'Nd', 'Sm'])]

VCCR_Glass_DyYb = VCCRREE[VCCRREE['variable'].isin(['Dy', 'Yb'])]
CCR_Glass_DyYb = MGREE[MGREE['variable'].isin(['Dy', 'Yb'])]



#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(8,5))


#group plot title
#title = fig.suptitle("Crystal-Rich (VCCR + CG) Fiamme Glass Populations", fontsize=18, y = 0.97)

#plot 1 

#create ree plot

#INITIAL VCCR AND CCR DATA!!

plot = sns.lineplot(data = MGREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief", ci = 'sd', linewidth = 1, dashes = True, style = 'Population')
#plot1 = sns.lineplot(data = VCCRREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], ci = 'sd', linewidth = 1, dashes = True, style = 'Population', style_order = ['VCCR 3', 'VCCR 1', 'VCCR 2'])

# plot = sns.lineplot(data = MGREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief", ci = 'sd', linewidth = 4, alpha = 0.05)
# plot1 = sns.lineplot(data = VCCRREE, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], ci = 'sd', linewidth = 4, alpha = 0.05)

# plot = sns.lineplot(data = VCCR_Glass_DyYb, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief", ci = 'sd', linewidth = 1)
# plot1 = sns.lineplot(data = CCR_Glass_DyYb, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], ci = 'sd', linewidth = 1)

# plot6 = sns.lineplot(data = VCCR_Glass_LaSm, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief", ci = 'sd', linewidth = 1)
# plot7 = sns.lineplot(data = CCR_Glass_LaSm, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], ci = 'sd', linewidth = 1)

 
#plot2 = sns.lineplot(data = melt_wr_LaCe, x= 'variable', y='value', hue = melt_wr_LaCe.index, sort = False, palette="Blues_d",legend="brief", ci = 'sd', linewidth = 1, dashes = True)


# plot2 = sns.scatterplot(data = MGREE_wr, x= 'variable', y='value', hue = 'Population', palette="Blues_d",legend=False, linewidth = 1, linestyle = " ")
# plot3 = sns.scatterplot(data = VCCRREE_wr, x= 'variable', y='value', hue = 'Population', palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], linestyle = " ")

# plot2 = sns.scatterplot(data = melt_wr, x= 'variable', y='value', hue = 'Population', palette="Blues_d",legend=False, linewidth = 1, )
# #plot3 = sns.scatterplot(data = VCCRREE_wr, x= 'variable', y='value', hue = 'Population', palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])


# plot2 = sns.lineplot(data = MGREE_wr, x= 'variable', y='value', hue = 'Population', sort = False, palette="Blues_d",legend="brief", ci = 'sd',linestyle = (0, (1, 10)), marker = 'o')
# plot3 = sns.lineplot(data = VCCRREE_wr, x= 'variable', y='value', hue = 'Population', sort = False, palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'], ci = 'sd', marker = 'o')

#plot2 = sns.scatterplot(data = MGREE_wr, x= 'variable', y='value', hue = 'Population', palette="Blues_d",legend=False)
#plot3 = sns.scatterplot(data = VCCRREE_wr, x= 'variable', y='value', hue = 'Population', palette="PuRd_r", hue_order = ['VCCR 1', 'VCCR 2', 'VCCR 3'])

#FG PLOT!
FGREE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/Supplementary_Data_Table_4_Normalized_REE.xlsx", sheet_name = 'I-FCP_Normalized')

#  change all negatives to NaN
num = FGREE._get_numeric_data()
num[num <= 0] = np.nan

FGREE = FGREE.dropna(axis=1, how = 'all')
FGREE = FGREE.dropna(axis=0, how = 'any')

#  change all negatives to NaN
num[num <= 0] = np.nan
FGmelt = (FGREE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
FG = FGmelt[FGmelt['Population'].isin(['ORA-5B-410', 'ORA-5B-412', 'ORA-5B-414'])]
FG = FG.replace(regex={'ORA-5B-410-B': 'ORA-5B-410', 'ORA-5B-412A-FG': 'ORA-5B-412', 'ORA-5B-414-FG': 'ORA-5B-414'})

#plot = sns.lineplot(data = FG, alpha = 0.3, linewidth = 2, zorder = 0, x= 'variable', y='value', hue = 'Sample', sort = False, palette="OrRd_r",legend="brief", hue_order = ('ORA-5B-412', 'ORA-5B-410', 'ORA-5B-414'), ci = 'sd')


# O-FCP DATA
# Add in REE
#import xlsx file
REE = pd.read_excel("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Trace_Avgs_NormalizedREE.xlsx", sheet_name = 'FGCP_Normalized')

#na_values = ['nan']

#  change all negatives to NaN
num = REE._get_numeric_data()
num[num <= 0] = np.nan

#REE = REE.dropna(axis=1, how = 'all')

REE = REE.dropna(axis=1, how = 'all')

REE = REE.dropna(axis=0, how = 'any')


#REE = REE.dropna(axis=0, how = 'any')

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
melt = (REE.melt(id_vars=['Sample','Population'], value_vars=['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'], ignore_index=False))
#melt = melt.set_index('Sample')

#melt = melt.set_index('Sample')
#melt = melt.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7','ORA-5B-412B-CG'], axis= 0)
#melt = melt.drop(['ORA-5B-405', 'ORA-5B-416'], axis= 0)
#melt = melt.reset_index()

# Dataframe Slicing using "isin"
ORA_002_REE = melt[melt['Population'].isin(['ORA-2A-002'])]
ORA_024_REE = melt[melt['Population'].isin(['ORA-2A-024'])]

ORA_002_REE = ORA_002_REE.replace(regex={'ORA-2A-002-Type1': 'ORA-2A-002-Type 1', 'ORA-2A-002-Type2': 'ORA-2A-002-Type 2', 'ORA-2A-002-Type3': 'ORA-2A-002-Type 3'})
ORA_024_REE = ORA_024_REE.replace(regex={'ORA-2A-024-TYPE1': 'ORA-2A-024-Type 1','ORA-2A-024-TYPE2': 'ORA-2A-024-Type 2' ,'ORA-2A-024-TYPE3': 'ORA-2A-024-Type 3','ORA-2A-024-TYPE4': 'ORA-2A-024-Type 4'})

error_config = {'alpha' : 0.05}
plot = sns.lineplot(data = ORA_002_REE, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens_d",legend="brief", alpha = 0.5, hue_order = ('ORA-2A-002-Type 1', 'ORA-2A-002-Type 2','ORA-2A-002-Type 3'), ci = 'sd', err_kws = error_config)
plot1 = sns.lineplot(data = ORA_024_REE, x= 'variable', y='value', hue = 'Sample', sort = False, palette="Greens_d",legend="brief", ci = 'sd', err_kws = error_config,  linewidth = 1, alpha = 0.5)


#-------
#Back to plotting whole rock data

plot2 = sns.scatterplot(data = CCR_LaSm, x= 'variable', y='value', hue = 'Population', palette="Blues_d",legend=False, s=100)
plot3 = sns.scatterplot(data = VCCR_LaSm, x= 'variable', y='value', hue = 'Population', palette="PuRd_r", hue_order = ['VCCR 2','VCCR 1', 'VCCR 3'], s = 100, legend = False)

plot4 = sns.scatterplot(data = CCR_DyYb, x= 'variable', y='value', hue = 'Population', palette="Blues_d",legend=False, s = 100)
plot5 = sns.scatterplot(data = VCCR_DyYb, x= 'variable', y='value', hue = 'Population', palette="PuRd_r", hue_order = ['VCCR 2', 'VCCR 1', 'VCCR 3'], s = 100, legend = False)

plot2 = sns.lineplot(data = CCR_LaSm, x= 'variable', y='value', hue = 'Population', palette="Blues_d",legend=False)
plot3 = sns.lineplot(data = VCCR_LaSm, x= 'variable', y='value', hue = 'Population', palette="PuRd_r", hue_order = ['VCCR 2', 'VCCR 1', 'VCCR 3'])

plot4 = sns.lineplot(data = CCR_DyYb, x= 'variable', y='value', hue = 'Population', palette="Blues_d",legend='brief')
plot5 = sns.lineplot(data = VCCR_DyYb, x= 'variable', y='value', hue = 'Population', palette="PuRd_r", hue_order = ['VCCR 2','VCCR 1', 'VCCR 3'])


plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)

#set location of legend
plt.legend(loc='lower right')

h,l = plot.get_legend_handles_labels()

#h,l = plot5.get_legend_handles_labels()

#plot just populations
#l[1:4] = ('CCR 1', 'CCR 2', 'CCR 3')
plt.legend(h[1:4]+h[5:9],l[1:4]+l[5:9],loc='lower right', fontsize = 15, ncol = 1, handlelength = 1.5, markerscale = 10)

#Plot for FG Data
#l[1:4] = ('CCR 1', 'CCR 2', 'CCR 3')
plt.legend(h[1:4]+h[5:8]+ l[10:],l[1:4]+l[5:8]+l[10:],loc='lower right', fontsize = 15, ncol = 1, handlelength = 1.5, markerscale = 10)

plt.xlabel('')
plt.ylabel('Sample/Chondrite', fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=18.5)

#plot1.text(-0.5,0.45, str('error envelopes $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')
plot.text(-0.5,0.12, str('error envelopes $\pm$ 1 std'), fontsize = 18.5, fontweight = 'normal')

#plt.ylim(0.05, 200)

#set y axis to log scale
plot.set(yscale='log')
plt.ylim( (10**-1,10**2.2) )

plt.ylim( (10**-1,10**2.6) )


for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

for tick in plot.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(25)

# Configure legend
#h, l = plot.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)


l[0] = "Outflow"
l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


# set size of plot
plt.tight_layout(pad = 0.75)
#plt.show()

# set size of plot
#sns.set_context("poster")


#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/all_wholerockREE.png', dpi=800)
