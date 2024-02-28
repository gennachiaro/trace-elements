#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


sns.set()

##FROM BA-SR-ALL.PY
# Specify pathname
path = '/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Standard_Data_Final_2023.xlsx'
#path = os.path.normcase(path) # This changes path to be compatible with windows

# Create a custom list of values I want to cast to NaN, and explicitly
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

# Columns converted to np.float with a dictionary
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# Master spreadsheet with clear mineral analyses removed (excel file!)
df = pd.read_excel(path, dtype = data, na_values = na_values)

#Drop included column
df = df.drop('Included', axis = 1)

# NaN treatment:
#   change all negatives and zeroes to NaN
num = df._get_numeric_data()
num[num <= 0] = np.nan

# Change analysis date to a string
s = df['Date']
df['Date'] = (s.dt.strftime('%Y.%m.%d'))

df = df.sort_values('Date')

sample_mean = df.groupby(
    ['Sample', 'Population', 'Date']).mean()
sample_mean = sample_mean.reset_index()

# Add in a column that tells how many samples were calculated for the mean using value_counts
count = df['Sample'].value_counts() #can use .size() but that includes NaN values

# Calculate stdev for each sample (messy)
sample_std = df.groupby(
    ['Sample', 'Population', 'Date']).std()
sample_std = sample_std.reset_index()

# Add in a column that tells how many samples were calculated for the stdev
sample_std = sample_std.set_index('Sample')
sample_std['Count'] = count

sample_std = sample_std.reset_index()

#DataFrameMelt to get all values for each spot in tidy data
    # every element for each spot corresponds to a separate row
        # this is for if we want to plot every single data point

#   Measured Standard Values:
#   NIST-610
#melt = (df.melt(id_vars=['Sample','Population', 'Date'], value_vars=['Li', 'Mg', 'Sc','Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   NIST-612
#melt = (df.melt(id_vars=['Sample','Population', 'Date'], value_vars=['Li', 'Mg', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   NIST-614
#melt = (df.melt(id_vars=['Sample','Population', 'Date'], value_vars=['Li', 'Mg', 'Sc','Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   RGM
melt = (df.melt(id_vars=['Sample','Population', 'Date'], value_vars=['Li', 'Mg', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Yb', 'Lu', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   ATHO-G
#melt = (df.melt(id_vars=['Sample','Population', 'Date'], value_vars=['Li','Sc', 'V', 'Cr', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))


#sample_mean = (sample_mean.melt(id_vars=['Sample','Population', 'Date'], value_vars=['Li', 'Mg', 'Al', 'Si', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#melt = melt.set_index('Sample')

#Import accepted values
# Columns converted to np.float with a dictionary
data = ({'Li': np.float64, 'Mg': np.float64, 'Al': np.float64,'Si': np.float64,'Ca': np.float64,'Sc': np.float64,'Ti': np.float64,'Ti.1': np.float64,'V': np.float64, 'Cr': np.float64, 'Mn': np.float64,'Fe': np.float64,'Co': np.float64,'Ni': np.float64, 'Zn': np.float64,'Rb': np.float64,'Sr': np.float64,'Y': np.float64,'Zr': np.float64,'Nb': np.float64,'Ba': np.float64,'La': np.float64,'Ce': np.float64,'Pr': np.float64,'Nd': np.float64,'Sm':np.float64,'Eu': np.float64,'Gd': np.float64,'Tb': np.float64,'Gd.1': np.float64,'Dy': np.float64,'Ho': np.float64,'Er': np.float64,'Tm': np.float64,'Yb': np.float64,'Lu':np.float64,'Hf': np.float64,'Ta': np.float64,'Pb': np.float64,'Th': np.float64,'U': np.float64})

# Master spreadsheet with clear mineral analyses removed (excel file!)
accepted = pd.read_excel(path, sheet_name = "Accepted_Values", dtype = data, na_values = na_values)

#   Accepted Melt Values:
#   NIST-610
#melt_a = (accepted.melt(id_vars=['Sample'], value_vars=['Li', 'Mg', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   NIST-612
#melt_a = (accepted.melt(id_vars=['Sample'], value_vars=['Li', 'Mg', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   NIST-614
#melt_a = (accepted.melt(id_vars=['Sample'], value_vars=['Li', 'Mg', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   RGM
melt_a = (accepted.melt(id_vars=['Sample'], value_vars=['Li', 'Mg', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Yb', 'Lu', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#   ATHO-G
#melt_a = (accepted.melt(id_vars=['Sample'], value_vars=['Li', 'Sc', 'V', 'Cr','Co', 'Ni', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U'], ignore_index=False))

#Slice by Sample

NIST_610 = sample_mean[sample_mean['Sample'].isin(['NIST-610'])]
NIST_612 = sample_mean[sample_mean['Sample'].isin(['NIST-612'])]
NIST_614 = sample_mean[sample_mean['Sample'].isin(['NIST-614'])]
RGM_1 = sample_mean[sample_mean['Sample'].isin(['RGM-1'])]
ATHO_G = sample_mean[sample_mean['Sample'].isin(['ATHO-G'])]

NIST_610_new = NIST_610[NIST_610['Date'].isin(['2023.03.29'])]
NIST_612_new = NIST_612[NIST_612['Date'].isin(['2023.03.29'])]
NIST_614_new = NIST_614[NIST_614['Date'].isin(['2023.03.29'])]
RGM_1_new = RGM_1[RGM_1['Date'].isin(['2023.03.29'])]

NIST_610_std = sample_std[sample_std['Sample'].isin(['NIST-610'])]
NIST_612_std = sample_std[sample_std['Sample'].isin(['NIST-612'])]
NIST_614_std = sample_std[sample_std['Sample'].isin(['NIST-614'])]
RGM_1_std = sample_std[sample_std['Sample'].isin(['RGM-1'])]
ATHO_G_std = sample_std[sample_std['Sample'].isin(['ATHO-G'])]

NIST_610_new_std = NIST_610_std[NIST_610_std['Date'].isin(['2023.03.29'])]
NIST_612_new_std = NIST_612_std[NIST_612_std['Date'].isin(['2023.03.29'])]
NIST_614_new_std = NIST_614_std[NIST_614_std['Date'].isin(['2023.03.29'])]
RGM_1_new_std = RGM_1_std[RGM_1_std['Date'].isin(['2023.03.29'])]

NIST_610 = NIST_610.set_index('Sample')
NIST_612 = NIST_612.set_index('Sample')
NIST_614 = NIST_614.set_index('Sample')
RGM_1 = RGM_1.set_index('Sample')
ATHO_G = ATHO_G.set_index('Sample')

NIST_610_new = NIST_610_new.set_index('Sample')
NIST_612_new = NIST_612_new.set_index('Sample')
NIST_614_new = NIST_614_new.set_index('Sample')
RGM_1_new = RGM_1_new.set_index('Sample')

NIST_610_std = NIST_610_std.set_index('Sample')
NIST_612_std = NIST_612_std.set_index('Sample')
NIST_614_std = NIST_614_std.set_index('Sample')
RGM_1_std = RGM_1_std.set_index('Sample')
ATHO_G_std = ATHO_G_std.set_index('Sample')

NIST_610_new_std = NIST_610_new_std.set_index('Sample')
NIST_612_new_std = NIST_612_new_std.set_index('Sample')
NIST_614_new_std = NIST_614_new_std.set_index('Sample')
RGM_1_new_std = RGM_1_new_std.set_index('Sample')

# # Write summary statistics to excel sheet

# with pd.ExcelWriter("/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/LAICPMS_Standards_Final.xlsx") as writer:
#     NIST_610.to_excel(writer, sheet_name = "NIST-610")
#     NIST_612.to_excel(writer, sheet_name = "NIST-612")
#     NIST_614.to_excel(writer, sheet_name = "NIST-614")
#     RGM_1.to_excel(writer, sheet_name = "RGM-1")
#     ATHO_G.to_excel(writer, sheet_name = "ATHO-G")
#     NIST_610_std.to_excel(writer, sheet_name = "NIST-610_std")
#     NIST_612_std.to_excel(writer, sheet_name = "NIST-612_std")
#     NIST_614_std.to_excel(writer, sheet_name = "NIST-614_std")
#     RGM_1_std.to_excel(writer, sheet_name = "RGM-1_std")
#     ATHO_G_std.to_excel(writer, sheet_name = "ATHO-G_std")

NIST_610 = melt[melt['Sample'].isin(['NIST-610'])]
NIST_612 = melt[melt['Sample'].isin(['NIST-612'])]
NIST_614 = melt[melt['Sample'].isin(['NIST-614'])]
RGM_1 = melt[melt['Sample'].isin(['RGM-1'])]
ATHO_G = melt[melt['Sample'].isin(['ATHO-G'])]

NIST_610_new = NIST_610[NIST_610['Date'].isin(['2023.03.29'])]
NIST_612_new = NIST_612[NIST_612['Date'].isin(['2023.03.29'])]
NIST_614_new = NIST_614[NIST_614['Date'].isin(['2023.03.29'])]
RGM_1_new = RGM_1[RGM_1['Date'].isin(['2023.03.29'])]

NIST_610 = NIST_610.set_index('Sample')
NIST_612 = NIST_612.set_index('Sample')
NIST_614 = NIST_614.set_index('Sample')
RGM_1 = RGM_1.set_index('Sample')
ATHO_G = ATHO_G.set_index('Sample')

NIST_610_new = NIST_610_new.set_index('Sample')
NIST_612_new = NIST_612_new.set_index('Sample')
NIST_614_new = NIST_614_new.set_index('Sample')
RGM_1_new = RGM_1_new.set_index('Sample')

# Slice accepted values
NIST_610_a = melt_a[melt_a['Sample'].isin(['NIST-610'])]
NIST_612_a = melt_a[melt_a['Sample'].isin(['NIST-612'])]
NIST_614_a = melt_a[melt_a['Sample'].isin(['NIST-614'])]
RGM_1_a = melt_a[melt_a['Sample'].isin(['RGM-1'])]
ATHO_G_a = melt_a[melt_a['Sample'].isin(['ATHO-G'])]

# NIST_610_a = NIST_610_a.set_index('Sample')
# NIST_612_a = NIST_612_a.set_index('Sample')
# NIST_614_a = NIST_614_a.set_index('Sample')
# RGM_1_a = RGM_1_a.set_index('Sample')
# ATHO_G_a = ATHO_G_a.set_index('Sample')

fig = plt.figure(figsize=(12,7))

# Set background color
sns.set_style("darkgrid")

title = fig.suptitle("RGM", fontsize=18, y = 0.95)

plot = sns.lineplot(data = RGM_1, x= 'variable', y='value', hue = 'Date', sort = False, palette="Blues_d",legend="brief", ci = 'sd')
plot1 = sns.lineplot(data = RGM_1_a, x= 'variable', y='value', hue = 'Sample', sort = False, ci = 'sd', legend = "brief", palette="Reds_d")
plot2 = sns.lineplot(data = RGM_1_new, x= 'variable', y='value', sort = False, ci = 'sd', legend = "brief", color = "yellow")

plot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
plot.grid(b=True, which='major', color='w', linewidth=1.0)
plot.grid(b=True, which='minor', color='w', linewidth=0.5)


h,l = plot.get_legend_handles_labels()
#h1,l1 = plot1.get_legend_handles_labels()

#plot just populations
#plt.legend(h[1:4]+h[5:9],l[1:4]+l[5:9],loc='lower right')
#plt.legend(h[1:9]+ h[10:], l[1:9] + l[10:], loc='lower center', ncol = 3)

#plt.legend(h1[1:], l1[1:], loc='lower center', ncol = 3)

plt.xlabel('')
plt.ylabel('')

#   Specific Plot Parameters:

#   Miscellaneous Parameters:
#   set y axis to log scale
#plot.set(yscale='log')
#plt.ylim( (10**1,10**100) )

#   NIST-610 Parameters:
# plot1.text(-1,328, str('error envelopes $\pm$ 1 std'), fontsize = 11, fontweight = 'normal')
# plt.legend(h[1:10]+ h[11:], l[1:10] + l[11:], loc='lower center', ncol = 3)
# plt.ylim( (325,525) )

#   NIST-612 Parameters:
# plot1.text(-1,22, str('error envelopes $\pm$ 1 std'), fontsize = 11, fontweight = 'normal')
# plt.legend(h[1:11]+ h[11:], l[1:10] + l[11:], loc='lower center', ncol = 3)

#   NIST-614 Parameters:
# plot1.text(-1,-2, str('error envelopes $\pm$ 1 std'), fontsize = 11, fontweight = 'normal')
# plt.legend(h[1:9]+ h[10:], l[1:9] + l[10:], loc='upper right', ncol = 3)

#   RGM Parameters:
l[11] = 'RGM-1 Values'
plot1.text(-1,-80, str('error envelopes $\pm$ 1 std'), fontsize = 11, fontweight = 'normal')
plt.legend(h[1:9]+ h[10:], l[1:9] + l[10:], loc='upper center', ncol = 3)
plt.ylim( (-100,1900) )


#   ATHO-G Parameters:
#l[7] = 'ATHO-G Values'
#plt.legend(h[1:6]+ h[7:], l[1:6] + l[7:], loc='upper center', ncol = 2)
#plot1.text(-1,-50, str('error envelopes $\pm$ 1 std'), fontsize = 11, fontweight = 'normal')
#plt.ylim( (-70,700) )


for axis in [plot.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)

plt.tight_layout()

#plt.savefig('/Users/gennachiaro/Documents/vanderbilt/research/writing/Ora Fiamme Paper 2021/Supplementary Info/NIST-610.png', dpi=800)

#plt.savefig('/Users/gennachiaro/Desktop/Weird_RGM2.png', dpi=800)
