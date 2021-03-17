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
tr = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/TraceElements_NoFilter.csv', index_col=1)

#FGCP = df.loc[['ORA-2A-002_Type1','ORA-2A-002_Type2','ORA-2A-002','ORA-2A-003','ORA-2A-016_Type1','ORA-2A-016-Type2','ORA-2A-016-Type3','ORA-2A-016-Type4','ORA-2A-023','ORA-2A-024','MINGLED1-ORA-2A-024','MINGLED2-ORA-2A-024','MINGLED3-ORA-2A-024']]
#MG = df.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]

#VCCR = tr.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]

VCCR = tr.loc[['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7']]
VCCR2 = tr.loc[['ORA-5B-414-CG', 'ORA-5B-412B-CG', 'ORA-5B-415', 'ORA-5B-412A-CG', 'ORA-5B-408-SITE8', 'ORA-5B-413', 'ORA-5B-411', 'ORA-5B-417']]
VCCR3 = tr.loc[['ORA-5B-416', 'ORA-5B-409']]

#VCCR = tr.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]

VCCR_index = VCCR.index
VCCR2_index = VCCR2.index
VCCR3_index = VCCR3.index


#slice data frame for individual sample
ORA5B402 = tr.loc[['ORA-5B-402']]

#get individual sample means 
print(ORA5B402.mean())

#get individual sample std dev
ORA5B402 = tr.loc[['ORA-5B-402']]


ORA5B402.std()
print(ORA5B402.std())
print(ORA5B402.mean())

#create dataframe of means and std devs
stats = pd.DataFrame({'Sample': VCCR_index},



#have to find a way to index the values and then transpose them 

#then slice the data frame by sample to get sample std dev and means

                        'Element': VCCR.loc[[0:1]]})

print(stats)
