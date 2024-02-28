#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
sns.set()

##FROM BA-SR-ALL.PY
# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-MASTER_2023.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

# Columns converted to np.float with a dictionary
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# Master spreadsheet with clear mineral analyses removed (excel file!)
df = pd.read_excel(path, sheet_name = 'Data', skiprows = [1], dtype = data, na_values = na_values)

#If Value in included row is equal to zero, drop this row
df = df.loc[~((df['Included'] == 0))]

#Drop included column
df = df.drop('Included', axis = 1)

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
df = df.dropna(axis=1, how='all')

# NaN treatment:
#   change all negatives and zeroes to NaN
num = df._get_numeric_data()
num[num <= 0] = np.nan

#   change all negatives to NaN
#num = df1._get_numeric_data()
#num[num < 0] = np.nan

#   drop rows with any NaN values
#df = df.dropna()
#df1 = df1.dropna()

#   fill NaN
#df = df.fillna(method = 'bfill')
#df1 = df1.fillna(method = 'bfill')

# Change analysis date to a string
s = df['Date']
df['Date'] = (s.dt.strftime('%Y.%m.%d'))

# Dropping Values:
df = df.set_index('Sample')

#Dropping MG samples because we remeasured these
df = df.drop(['ORA-2A-036-B','ORA-2A-036','ORA-2A-032','ORA-2A-035'], axis= 0)

#Dropping VCCR samples because we remeasured these
df = df.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B', 'ORA-5B-416-B'], axis= 0)

#ADDED THIS RECENTLY
#Drop VCCR samples because they are the same fiamma:
df = df.drop(['ORA-5B-405', 'ORA-5B-416'])

# Dropping VCCR samples because we don't have matching SEM values
df = df.drop(['ORA-5B-408-SITE8', 'ORA-5B-408-SITE7', 'ORA-5B-412B-CG'], axis= 0)

# Dropping FG samples because remeasured:
df = df.drop(['ORA-5B-410','ORA-5B-412B-FG'], axis= 0)
df = df.reset_index()

#---------
# Calculate means for each sample (messy)
# sample_mean = df.groupby('Sample').mean()

sample_mean = df.groupby(
    ['Sample', 'Population', 'Date']).mean()
sample_mean = sample_mean.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df['Sample'].value_counts() #can use .size() but that includes NaN values

sample_mean = sample_mean.set_index('Sample')
sample_mean['Count'] = count

# Calculate stdev for each sample (messy)
sample_std = df.groupby(
    ['Sample', 'Population', 'Date']).std()
sample_std = sample_std.reset_index()

# Add in a column that tells how many samples were calculated for the stdev
sample_std = sample_std.set_index('Sample')
sample_std['Count'] = count

sample_std = sample_std.reset_index()
sample_mean = sample_mean.reset_index()

# Dataframe Slicing of average values using "isin"
MG = sample_mean[sample_mean['Population'].isin(['MG 1'])]
DEVIT = sample_mean[sample_mean['Population'].isin(['DEVIT'])]
DEVIT_All = df[df['Population'].isin(['DEVIT'])]

# #FGCP = FGCP.drop(['ORA-2A-002'], axis = 0)

# Multiply dataframe by two to get 2 sigma
#sample_std = sample_std *2
# print(sample_std.head())

# Set index
sample_std = sample_std.set_index('Sample')
#print (sample_std.head())

# Select sample stdev by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1'])]
DEVIT_std = sample_std[sample_std['Population'].isin(['DEVIT'])]


# Plotting
#       Slicing dataframe

# Set background color
sns.set_style("darkgrid")

# Set axes limits
#plt.ylim(10, 50)
#plt.xlim (-0.2,0.4)

# Set color palette
# sns.set_palette("PuBuGn_d")


# plt.legend(h[0:5]+h[5:14],l[0:4]+l[5:14], loc='best')



# Set size of plot
#sns.set_context("paper") 


#plt.savefig('/Users/gennachiaro/Desktop/2023_SrBa.png', dpi=600, bbox_inches = 'tight')


fig = plt.figure(figsize=(11,8.5))



# Plotting
# Select elements to plot
x = 'Ba'
y = 'Sr'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = DEVIT_std[x]
yerr2 = DEVIT_std[y]

# Create plot
plt.subplot(2,2,1)

#Added style to the plot!
plot = sns.scatterplot(data=DEVIT, x=x, y=y, hue="Sample", palette="Reds", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, legend= 'brief', alpha=0.8)
plt.errorbar(x=DEVIT[x], y=DEVIT[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='red', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plot = sns.scatterplot(data=DEVIT_All, x=x, y=y, hue="Sample", palette="Reds", markers=('X','o','s'), style = "Sample",
                       edgecolor="black", s=200, legend= False, alpha=0.2)

plot = sns.scatterplot(data=MG, x=x, y=y, hue="Sample", palette="Blues_d", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, alpha=0.8, legend= 'brief')
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)


plot.text(3.5,-1, str('error bars $\pm$ 1s'), fontsize = 18.5, fontweight = 'normal')

# # set location of legend
plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

# Turn on Log Axes

h, l = plot.get_legend_handles_labels()

l[0] = "Devitrified Glass"
l[1] = 'ORA-2A-001'
l[2] = 'ORA-2A-031'
l[3] = 'ORA-2A-040'
l[4] = "Non-Devitrified Glass"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 1, fontsize = 15, markerscale = 1.6)


plt.legend(h, l, loc='lower right', bbox_to_anchor=(1.75, -2), ncol=2, handlelength = 1, columnspacing = 1, fontsize = 15, markerscale = 1.6)
#plt.legend(h[0:1], l[0:1], loc='lower right', ncol=4, handlelength = 1, columnspacing = 1, fontsize = 15, markerscale = 1.6)

# h, l = plot.get_legend_handles_labels()
# # plt.legend(h[1:7]+h[7:10]+h[11:14]+h[23:26], l[1:7]+l[7:10]+l[11:14] +
# #            l[23:26], loc='lower right', bbox_to_anchor=(2, -3), ncol=5, fontsize=11)

# plt.legend(h [1:4] + h[5:], l[1:4] + l[5:],loc='center left', bbox_to_anchor=(1.18, 0.5), ncol=2, fontsize = "x-large", markerscale = 1.5, title = "Devitrified Glass          Non-Devitrified Glass", title_fontsize = "x-large",  handlelength = 1, columnspacing = 2.5)



#plot 2
plt.subplot(2,2,2)
#create trace element plot

# Select elements to plot
x = 'Rb'
y = 'Hf'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = DEVIT_std[x]
yerr2 = DEVIT_std[y]

# Create plot
plt.subplot(2,2,2)

#Added style to the plot!
plot2 = sns.scatterplot(data=DEVIT, x=x, y=y, hue="Sample", palette="Reds", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, legend= False, alpha=0.8)
plt.errorbar(x=DEVIT[x], y=DEVIT[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='red', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plot = sns.scatterplot(data=DEVIT_All, x=x, y=y, hue="Sample", palette="Reds", markers=('X','o','s'), style = "Sample",
                       edgecolor="black", s=200, legend= False, alpha=0.2)

plot2 = sns.scatterplot(data=MG, x=x, y=y, hue="Sample", palette="Blues_d", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, alpha=0.8, legend= False)
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)


plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)


#plot 3
plt.subplot(2,2,3)

x = 'Lu'
y = 'La'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = DEVIT_std[x]
yerr2 = DEVIT_std[y]

# Create plot
plt.subplot(2,2,3)

#Added style to the plot!
plot3 = sns.scatterplot(data=DEVIT, x=x, y=y, hue="Sample", palette="Reds", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, legend= False, alpha=0.8)
plt.errorbar(x=DEVIT[x], y=DEVIT[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='red', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plot = sns.scatterplot(data=DEVIT_All, x=x, y=y, hue="Sample", palette="Reds", markers=('X','o','s'), style = "Sample",
                       edgecolor="black", s=200, legend= False, alpha=0.2)

plot3 = sns.scatterplot(data=MG, x=x, y=y, hue="Sample", palette="Blues_d", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, alpha=0.8, legend= False)
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

#plot3.text(28,11, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# Configure legend
#h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


#plot 4
plt.subplot(2,2,4)
x = 'Nd'
y = 'Y'

xerr1 = MG_std[x]
yerr1 = MG_std[y]

# VCCR Error Bar Values
xerr2 = DEVIT_std[x]
yerr2 = DEVIT_std[y]

# Create plot
plt.subplot(2,2,4)

#Added style to the plot!
plot4 = sns.scatterplot(data=DEVIT, x=x, y=y, hue="Sample", palette="Reds", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, legend= False, alpha=0.8)
plt.errorbar(x=DEVIT[x], y=DEVIT[y], xerr=xerr2, yerr=yerr2, ls='none',
             ecolor='red', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)

plot = sns.scatterplot(data=DEVIT_All, x=x, y=y, hue="Sample", palette="Reds", markers=('X','o','s'), style = "Sample",
                       edgecolor="black", s=200, legend= False, alpha=0.2)

plot4 = sns.scatterplot(data=MG, x=x, y=y, hue="Sample", palette="Blues_d", markers=('o','X','s'), style = "Sample",
                       edgecolor="black", s=250, alpha=0.8, legend= False)
plt.errorbar(x=MG[x], y=MG[y], xerr=xerr1, yerr=yerr1, ls='none',
             ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

plt.xlabel(x + ' [ppm]', fontsize = 18.5)
plt.ylabel(y + " [ppm]", fontsize = 18.5)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

#plot4.text(34.5,17, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')


# Configure legend
h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)


# l[0] = "Outflow"
# l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)


# set size of plot
#plt.tight_layout(pad = 3.0)
#plt.subplots_adjust
#fig.tight_layout(pad = 3.0)

# ADD IN EXTRA SPACE FOR LOG PLOT
plt.subplots_adjust(hspace = 0.25, wspace = 0.25)
#plt.tight_layout(pad = 0.75)

plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/graphs/DEVIT_4Plot_ErrorBars_Revision2.svg', dpi=800)
