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
from PIL.Image import merge

# import csv file
# all values
#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv')

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
#df1 = df1.dropna(axis = 1, how = 'all')



#indexed dataframe for 
df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', index_col=1)

df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv')

# drop blank columns
df = df.dropna(axis = 1, how = 'all')
df1 = df1.dropna(axis = 1, how = 'all')

# drop rows with any NaN values
#df = df.dropna()
#df1 = df1.dropna()

# fill NaN with zeroes
df = df.fillna(method = 'bfill')
df1 = df1.fillna(method = 'bfill')

#df.iloc[3:62][df.iloc[3:62]< 0] = 0

#df = np.where(df<0, 0, df)

num = df._get_numeric_data()
num[num < 0] = 0

num = df1._get_numeric_data()
num[num < 0] = 0


#df.clip(lower=0)

#map out columns
col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df.columns)]


#df['Li'] = df['Li'].apply(lambda x : x if x > 0 else 0)


#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
#FGCP_index = FGCP.index

#MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#MG_index = MG.index

#VCCR = df.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
#VCCR_index = VCCR.index

#VCCR1 = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#VCCR1_index = VCCR1.index

#MG1 = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
#MG1_index = MG1.index


#DataFrameMelt
melt = (df.melt(id_vars=['Spot', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))
print(melt)

#stats = melt.groupby(['Sample', 'Population', 'variable'])['value'].describe().reset_index()


#Get the mean values for each sample

#ERROR, NOT GETTING Li, Mg, AND MANY OTHER VALUES!!!!!!!!!!!!!!!!

#mean = df.mean(level = 'Sample')

mean = df.iloc[:,0:61].mean(level = "Sample")

#mean = df.mean(level = 'Sample', skipna = True)

#mean = df1.groupby('Sample').mean().reset_index()

#mean = df1.groupby(['Sample', 'Population'])[['Sr']].mean().reset_index()
#mean = df1.groupby(['Sample', 'Population']).mean().reset_index()


#mean.fillna(0)

print(mean.head())

#drop blank columns
#mean.dropna(axis = 1, how = 'all')


#print(mean)

#Create dataframe with sample populations
populations = df1[['Sample','Population']].drop_duplicates('Sample')


# Merge two dataframes

#merge = pd.merge(populations, mean, how = 'right', on= "Sample")
#merge = pd.merge(populations, mean, how = 'right', on= mean.index)

#merge = pd.merge(populations, mean, how = 'left', left_on= "Sample", right_on = "Sample")



#merge = pd.merge(populations, mean, how = 'right', left_on= "Sample", right_on = mean.index)

#print (merge.head())


# Append dataframe with population names




#DataFrameMelt for tidy data and plotting final values

#melt = (merge.melt(id_vars=['Sample', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr','SiO2'], ignore_index=False))
#print(melt)







#mean.join(populations["Population"])

#Testing
