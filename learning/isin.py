# IsIn Testing Code
#   Getting values by their population 

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


# Import csv file
    #All values
#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64}, na_values= na_values)

# Create a custom list of values I want to cast to NaN, and explicitly 
#   define the data types of columns:
na_values = ['<-1.00', '****', '<****', '*****']

#df = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv', index_col=1)
#df1 = pd.read_csv('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.csv',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values)
# read in excel file!

# All with clear mineral analyses removed
df1 = pd.read_excel('/Users/gennachiaro/Documents/vanderbilt/research/ora caldera/trace-elements/new-spreadsheets/Ora-Glass-All.xlsx',dtype={'Li': np.float64, 'Mg': np.float64, 'V': np.float64, 'Cr': np.float64, 'Ni': np.float64, 'Nb': np.float64,'SiO2': np.float64}, na_values= na_values, sheet_name = 'Data')

# Drop "Included" column
df1 = df1.drop(['Included'], axis = 1)

# drop blank columns
#df = df.dropna(axis = 1, how = 'all')
df1 = df1.dropna(axis = 1, how = 'all')

# NaN treatment:
#   change all negatives and zeroes to NaN
num = df1._get_numeric_data()
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

# map out columns
#col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df1.columns)]

# DataFrameMelt to get all values for each spot in tidy data
#   every element for each spot corresponds to a separate row
#       this is for if we want to plot every single data point
melt = (df1.melt(id_vars=['Sample','Spot', 'Population'], value_vars=['Li','Mg','Al','Si','Ca','Sc','Ti','Ti.1','V','Cr','Mn','Fe','Co','Ni','Zn','Rb','Sr','Y','Zr','Nb','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Gd.1','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U','Rb/Sr','Ba/Y','Zr/Y','Zr/Ce','Zr/Nb','U/Ce','Ce/Th','Rb/Th','Th/Nb','U/Y','Sr/Nb','Gd/Yb','U/Yb','Zr/Hf','Ba + Sr'], ignore_index=False))
#print(melt)

# Calculate means for each sample (messy)
sample_mean = df1.groupby('Sample').mean()
#print(sample_mean.head())

# Another way to calculate means, but need an indexed df to use "level"
#   we need to use a non-indexed dataframe to make our "populations" dataframe 
#       so might as well just use one dataframe and the groupby fx to make it simpler
#mean = df.mean(level = 'Sample') # another way to calculate means, but need an indexed df to use "level"

# Create seperate dataframe with sample populations
#   this is so we can merge this with the mean dataframe because the groupby fx just gives sample name and value for each element
populations = df1[['Sample','Population']].drop_duplicates('Sample')

# Merge two dataframes
merge_mean = pd.merge(populations, sample_mean, how = 'right', left_on= "Sample", right_on = sample_mean.index)

# Set index
merge_mean = merge_mean.set_index('Sample')
#print (merge.head())

# Calculate stdev for each sample (messy)
sample_std = df1.groupby('Sample').std()

# Multiply dataframe by two to get 2 sigma
#sample_std = sample_std *2
#print(sample_std.head())

# Merge two dataframes (stdev and populations)
sample_std = pd.merge(populations, sample_std, how = 'right', left_on= "Sample", right_on = sample_std.index)
#print (sample_std.head())

# Set index
sample_std = sample_std.set_index('Sample')
#print (sample_std.head())

# Plotting
#       Slicing dataframe
MG = merge_mean.loc[['ORA-2A-001','ORA-2A-005','ORA-2A-018','ORA-2A-031','ORA-2A-032','ORA-2A-035','ORA-2A-040']]
MG_index = MG.index

VCCR1 = merge_mean.loc [['ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-411','ORA-5B-416']]
VCCR1_index = VCCR1.index

#VCCR1 = df.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-406','ORA-5B-409','ORA-5B-411','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
#VCCR1_index = VCCR1.index

VCCR = merge_mean.loc [['ORA-5B-402','ORA-5B-404A','ORA-5B-404B','ORA-5B-405','ORA-5B-406','ORA-5B-407','ORA-5B-408-SITE2','ORA-5B-408-SITE7','ORA-5B-408-SITE8','ORA-5B-409','ORA-5B-411','ORA-5B-412A-CG','ORA-5B-412B-CG','ORA-5B-413','ORA-5B-414-CG','ORA-5B-415','ORA-5B-416','ORA-5B-417']]
VCCR_index = VCCR.index

VCCR_mean = merge_mean[merge_mean['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
VCCR_mean = VCCR_mean.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B','ORA-5B-416-B'], axis = 0)

print(VCCR.equals(VCCR_mean))

# Set background color
sns.set_style("darkgrid")

# Set axes limits
#plt.ylim(10, 50)
#plt.xlim (-0.2,0.4)

# Set color palette
#sns.set_palette("PuBuGn_d")

# Get error bar values

# Select MG standard samples by population
MG_std = sample_std[sample_std['Population'].isin(['MG 1', 'MG 2', 'MG 3'])]
# Drop bad samples
MG_std = MG_std.drop(['ORA-2A-004', 'ORA-2A-036'], axis = 0)

# Select VCCR standard samples by population
VCCR_std = sample_std[sample_std['Population'].isin(['VCCR 1', 'VCCR 2', 'VCCR 3'])]
# Drop bad samples
VCCR_std = VCCR_std.drop(['ORA-5B-405-B', 'ORA-5B-406-B','ORA-5B-409-B','ORA-5B-416-B'], axis = 0)
