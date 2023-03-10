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
import matplotlib as mpl
import scipy

# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/EDS-Laser_Test/EDS_Laser_Test_HRL-21.xlsx'

#path = os.path.normcase(path) # This changes path to be compatible with windows

# Master spreadsheet with clear mineral analyses removed (excel file!)
values = pd.read_excel(path, sheet_name = 'Values')
#errors = pd.read_excel(path, sheet_name = 'Std')

# drop blank rows
#df = df.dropna(axis = 1, how = 'all')

values = values.dropna(axis=0, how='any')

#   drop if there are any zeroes
values = values[~(values == 0).any(axis=1)]

Fifteen = values[values['Population'].isin(['15 Sec'])]

Sixty = values[values['Population'].isin(['60 Sec'])]


#set background color
sns.set_style("darkgrid")

#plot matrix
fig = plt.figure(figsize=(10,4))

# group plot title
#title = fig.suptitle("All Ora Fiamme Glass Trace Elements", fontsize=16, y=0.925)
title = fig.suptitle("Comparing EDS vs. LA-ICPMS Measurements (All)", fontsize=16, y=0.955)

#plot 1 

# Plotting
# Select elements to plot
x = 'Ti [ppm], EDS'
y = 'Ti [ppm], LA-ICPMS'


# Create plot
plt.subplot(1,2,1)

from scipy import stats
res = stats.linregress(values[x], values[y])
print(f"R-squared: {res.rvalue**2:.6f}")


p = sns.regplot(data = values, x = values[x],y = values[y], ci = None, color = 'green')

#plot1 = sns.scatterplot(data=values, x=x, y=y, hue="Name", palette="Blues_d", marker='s', style = "Population", edgecolor="black", s=150, alpha=0.8, legend= False)

plot1 = sns.scatterplot(data=values, x=x, y=y, hue="Name", palette="Blues_d", marker='s', style = "Name", edgecolor="black", s=150, alpha=0.8, legend= False)
# plot1 = sns.scatterplot(data=Sixty, x=x, y=y, hue="Name", palette="Blues_d", marker='s', style = "Name", edgecolor="black", s=150, alpha=0.8, legend= False)


# from statsmodels.formula.api import ols

# model1 = ols('values[x] ~ values[y]', data=values).fit(cov_type = 'HC3')
#plt.annotate('$R^2: {}$'.format(round(model1.rsquared_adj,3)), (500, 100)))

# plot1 = sns.scatterplot(data=Fifteen-Second, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
#                        edgecolor="black", s=150, alpha=0.8, legend= False)
# plt.errorbar(x=CG[x], y=CG[y], xerr= xerr1, yerr = yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# p = sns.regplot(data = CG, x = CG[x],y = CG[y], ci = None, color = 'cornflowerblue')

# slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
#                                                        y=p.get_lines()[0].get_ydata())

# print(intercept, slope)


# plot1 = sns.scatterplot(data=Sixty-Second, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
#                        edgecolor="black", s=150, legend= False, alpha=0.8)
# plt.errorbar(x=VCCR[x], y=VCCR[y], xerr= xerr2, yerr = yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


lims = [
    np.min([plot1.get_xlim(), plot1.get_ylim()]),  # min of both axes
    np.max([plot1.get_xlim(), plot1.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
plot1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#plot1.set_aspect('equal')
# plot1.set_xlim((300, 900))
# plot1.set_ylim(lims)

#15 sec

# All
plot1.set_xlim((-100, 1600))
plot1.set_ylim((-100, 1600))



#plot1.set_ylim((500,675))
# plot1.set_xlim(100,650)
# plot1.set_ylim(100,650)

#lims set with no error bars

# p = sns.regplot(data = VCCR, x = VCCR[x],y = VCCR[y], ci = None, color = 'palevioletred')

# slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
#                                                        y=p.get_lines()[0].get_ydata())

# print(intercept, slope)

#sns.regplot(data = values, x = values[x],y = values[y], ci = None)

#plot.text(40,0.12, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#plt.annotate(f"$R^2: {res.rvalue**2:.3f}$", (820, 80))

#60sec
#plt.annotate(f"$R^2: {res.rvalue**2:.3f}$", (700, 450))

#15sec
plt.annotate(f"$R^2: {res.rvalue**2:.3f}$", (1000, 100))


#plot 2
plt.subplot(1,2,2)

# Select elements to plot
x = 'Mg [ppm], EDS'
y = 'Mg [ppm], LA-ICPMS'

# Create plot
plt.subplot(1,2,2)

res = stats.linregress(values[x], values[y])
print(f"R-squared: {res.rvalue**2:.6f}")


sns.regplot(data = values, x = values[x],y = values[y], ci = None, color = 'green')

#plot2 = sns.scatterplot(data=values, x=x, y=y, hue="Name", palette="Blues_d", marker='s', style = "Population", edgecolor="black", s=150, alpha=0.8, legend= 'brief')
plot2 = sns.scatterplot(data=values, x=x, y=y, hue="Name", palette="Blues_d", marker='s', style = "Name", edgecolor="black", s=150, alpha=0.8, legend= 'brief')
# plot2 = sns.scatterplot(data=Sixty, x=x, y=y, hue="Name", palette="Blues_d", marker='s', style = "Name", edgecolor="black", s=150, alpha=0.8, legend= 'brief')

# model2 = ols('values[x] ~ values[y]', data=values).fit(cov_type = 'HC3')
# plt.annotate('$R^2: {}$'.format(round(model2.rsquared_adj,3)),
#             (500, 100))

# plot2 = sns.scatterplot(data=CG, x=x, y=y, hue="Population", palette="Blues_d", marker='s', style = "Population",
#                        edgecolor="black", s=150, alpha=0.8, legend='brief', hue_order=['CG 1', 'CG 2', 'CG 3'])
# plt.errorbar(x=CG[x], y=CG[y], xerr= xerr1, yerr = yerr1, ls='none',
#              ecolor='cornflowerblue', elinewidth=1, capsize=2, alpha=0.8)

# plot2 = sns.scatterplot(data=VCCR, x=x, y=y, hue="Population", palette="PuRd_r", markers = ('h','^','P'), style = "Population",
#                        edgecolor="black", s=150, legend='brief', alpha=0.8, hue_order=['VCCR 1', 'VCCR 2', 'VCCR 3'])
# plt.errorbar(x=VCCR[x], y=VCCR[y], xerr= xerr2, yerr = yerr2, ls='none',
#              ecolor='palevioletred', elinewidth=1, capsize=2, barsabove=False, alpha=0.8)


lims = [
    np.min([plot2.get_xlim(), plot2.get_ylim()]),  # min of both axes
    np.max([plot2.get_xlim(), plot2.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
plot2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#plot2.set_aspect('equal')

#60 sec
# plot2.set_xlim((0, 600))
# plot2.set_ylim(lims)


#15 sec
plot2.set_xlim((0, 700))
plot2.set_ylim((200, 400))


#60 sec
#plt.annotate(f"$R^2: {res.rvalue**2:.3f}$", (450, 120))

#15 sec
plt.annotate(f"$R^2: {res.rvalue**2:.3f}$", (500, 220))


#plot.text(40,0.12, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

#plot2.text(76,4.7, str('error bars $\pm$ 1$\sigma$'), fontsize = 11, fontweight = 'normal')

# Configure legend
#h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
#plt.legend(h[1:4]+h[5:8], l[1:4]+l[5:8], loc='best', ncol=2)

# l[0] = "Outflow"
# l[4] = "Intracaldera"


# Configure legend
h, l = plot2.get_legend_handles_labels()

# Legend outside of plot
#plt.legend(h, l,loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

#plt.legend(h[1:4]+h[5:8],l[1:4]+l[5:8],loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# Legend inside of plot
plt.legend(h[1:],l[1:],loc='upper left', ncol=1, handlelength = 1, columnspacing = 0.5)


# l[0] = "Outflow"
# l[4] = "Intracaldera"

#plt.legend(h, l, loc='best', ncol = 2, handlelength = 1, columnspacing = 0.5)
#plt.legend(h, l, loc='lower right', bbox_to_anchor=(2, -2.366), ncol=1, fontsize=11)


# set size of plot
#plt.tight_layout(pad = 1.0)


#plt.subplots_adjust
#fig.tight_layout(pad = 3.0)

# ADD IN EXTRA SPACE FOR LOG PLOT
#plt.subplots_adjust(hspace = 0.55)


#plt.show()

# set size of plot
#sns.set_context("poster")

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/EDS-Laser_Test/HRL_Plots/HRL_all.png', dpi=500)

