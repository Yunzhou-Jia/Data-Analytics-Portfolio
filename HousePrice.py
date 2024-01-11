#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:14:16 2023

@author: yun
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# =============================================================================
# Data Preparation
# =============================================================================
# Loading data from CSV files
df = pd.read_csv('/Users/yun/Desktop/Exercise/SydenyHousePrices/SydneyHousePrices.csv')
location_df=pd.read_csv('/Users/yun/Desktop/Exercise/SydenyHousePrices/sydney_suburbs.csv')
# Displaying initial data for a quick overview
print('Data preview:\n', df.head())
print(df.info(), '\nNumber of records: {0}\nNumber of features: {1}'.format(df.shape[0], df.shape[1]))
print(df.describe())

# Checking for missing and duplicate values
empty_values = df.isnull().sum()
count = pd.DataFrame()
for column in df.columns:
    unique_values_count = len(df[column].value_counts())
    count = count.append({'feature': column, 'total unique values': unique_values_count}, ignore_index=True)

# Outlier detection and visualization
plt.figure()
category_data = [df['bed'].dropna(), df['bath'].dropna(), df['car'].dropna()]
plt.boxplot(category_data, labels=['bed', 'bath', 'car'])
plt.title('Outlier Check')
plt.show()
plt.figure()
plt.boxplot(df['sellPrice'], labels=['sellPrice'])
plt.show()

# =============================================================================
# Findings:
# The dataset contains 199503 records and 8 features.
# Issues identified include incorrect data types for 'Date', 'bed', and 'car', 
# and missing values in 'bed' and 'car' columns.
# =============================================================================

# =============================================================================
# Data Cleaning
# =============================================================================
# Filtering data and fixing data types
df = df[df['propType']=='house']# Removing property type other than "house" 
df.drop(df[df['postalCode'] >= 3000].index, inplace=True)  # Removing non-Sydney postcodes
df = df.drop_duplicates()  # Removing duplicate entries
df = df.drop(columns=['Id'])  # Dropping the unneeded 'Id' column
df.rename(columns={'Date': 'date'}, inplace=True)  # Renaming columns for consistency
df = df.fillna(0)  # Filling missing values with 0

# Correcting data types
df['date'] = pd.to_datetime(df['date'])
df['bed'] = df['bed'].astype(int)
df['car'] = df['car'].astype(int)

# Removing outliers based on bedrooms, bathrooms, and car park areas
def box_outliers(data, feature, scale):
    """
    Function to remove outliers based on IQR.
    Args:
        data (DataFrame): The DataFrame to process.
        feature (str): The feature for outlier detection and removal.
        scale (float): The scale to calculate the IQR range.
    Returns:
        DataFrame: The DataFrame after outlier removal.
    """
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[feature] >= Q1 - (scale * IQR)) & (data[feature] <= Q3 + (scale * IQR))]

df = box_outliers(df, 'bed', 1.5)
df = box_outliers(df, 'bath', 1.5)
df = box_outliers(df, 'car', 1.5)

# Applying various methods to remove sell price outliers
IQR_df = box_outliers(df, 'sellPrice', 1.5)  # Method 1: Interquartile Range

# Percentile method to determine outlier thresholds
percentiles = [0.0001, 0.0005, 0.009, 0.01, 0.9, 0.99, 0.999]
for percentile in percentiles:
    value = df['sellPrice'].quantile(percentile)
    print(f"{percentile * 100:5,.2f} percentile: {value:2,.0f}")

lq = df.sellPrice.quantile(0.009)
uq = df.sellPrice.quantile(0.999)
percentile_df = df[(df.sellPrice >= lq) & (df.sellPrice <= uq)]

# Deciding to use the percentile method for outlier removal
df = percentile_df
# =============================================================================
# Based on the above comparision, I decided to remove outliers by percentile, which only removes 0.1% of data points, 
# comparing to IQR, which drops 5.6% of data and still keeps those points with sell price of $1
# Finally, data is cleaned, with , 175106 data points remaining. Next step, visualization and analysis.
# =============================================================================
# =============================================================================
# Analysis
# =============================================================================
# Investigating key aspects of the housing market such as most expensive locations, busiest months/years, and transaction volume changes
# Mean value is used for assessing location prices to get a representative measure in datasets devoid of outliers
# Analysis by suburb and postal code
suburb_price = df.groupby('suburb').agg({'sellPrice': 'mean', 'postalCode': 'count'}).reset_index().sort_values(by='sellPrice', ascending=False)
suburb_price.columns = ['Suburb', 'Mean Price', 'Number of Transactions']

postalCodePrice = df.groupby('postalCode').agg({'sellPrice': 'mean', 'suburb': 'count'}).reset_index().sort_values(by='sellPrice', ascending=False)
postalCodePrice.columns = ['Postal Code', 'Mean Price', 'Number of Transactions']
#color
blue = '#496595'
grey = '#c6ccd8'
# Preparing data for visualization
top_10_suburb = suburb_price.head(10).assign(color=blue)
top_10_postalCode = postalCodePrice.head(10).assign(color=blue)
top_10_suburb['color'][3:] = grey
top_10_postalCode['color'][3] = grey

# Visualization of the top 10 mean prices by suburb and postal code (currently commented out)
plt.figure()

ax1=plt.subplot(1,2,1)
ax1_y_ticks=np.arange(len(top_10_suburb['Suburb']))
ax2_y_ticks=np.arange(len(top_10_postalCode['Postal Code']))
ax1.barh(ax1_y_ticks,top_10_suburb['Mean Price'],align='center',color=top_10_suburb['color'])
ax1.invert_yaxis()
ax1.set_yticks(ax1_y_ticks)
ax1.set_yticklabels(top_10_suburb['Suburb'])
ax1.set_title('Top 10 Mean Price of House Suburb',fontsize=9)


ax2=plt.subplot(1,2,2)
ax2.barh(ax2_y_ticks,top_10_postalCode['Mean Price'],color=top_10_postalCode['color'])
ax2.invert_yaxis()
ax2.set_yticks(ax2_y_ticks)
ax2.set_yticklabels(top_10_postalCode['Postal Code'])
ax2.set_title('Top 10 Mean Price of House Postal Code',fontsize=9)
plt.show()

#2 the busiest month over the year
#Create the time features
df['year']=pd.DatetimeIndex(df['date']).year
df['month']=pd.DatetimeIndex(df['date']).month
df['month_name']=df['date'].dt.month_name().str[:3]
#create month dataframe
month_df=df.groupby(['month','month_name']).agg({'sellPrice':'mean','suburb':'count'}).reset_index()
month_df=month_df.sort_values(by='suburb',ascending=False)
month_df.columns=['month','month_name','Mean sell prices','Transactions']
#create year dataframe
year_df=df.groupby('year').agg({'sellPrice':'mean','suburb':'count'}).reset_index()
year_df=year_df.sort_values(by='suburb',ascending=False)
year_df.columns=['year','Mean sell prices','Transactions']
year_df2=df.groupby('year').agg({'sellPrice':'mean','suburb':'count'}).reset_index()
#set the chart color
month_df['color']=blue
month_df['color'][3:]=grey
year_df['color']=blue
year_df['color'][3:]=grey

#set the list
year_list=range(math.floor(year_df['year'].min()),math.ceil(year_df['year'].max()))
month_list=range(math.floor(month_df['month'].min()),math.ceil(month_df['month'].max()))

#Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
#plots the ax1
ax1.bar('month','Transactions',data=month_df,color=month_df['color'])
ax1.set_xlabel('month')
ax1.set_title('The busiest month')
ax1.set_xticks(month_list)

ax2.bar('year','Transactions',data=year_df,color=year_df['color'])
ax2.set_xlabel('year')
ax2.set_title('The busiest year')
ax2.set_xticks(year_list)
#plots the line chart of ax2
ax2.plot(year_df2['year'],year_df2['suburb'],color='r',marker='o')

# =============================================================================
# Merging and outputting the final dataset and use the Tableau for visualisation
# =============================================================================
# Combining the location dataset with the cleaned housing price data
merge_df = pd.merge(location_df.rename(columns={'Suburb': 'suburb'}), df, on='suburb')
# Optionally outputting the dataset to a CSV file
# merge_df.to_csv('/Users/yun/Desktop/Exercise/SydneyHousePrices/sydneyHousingPrice.csv', index=False)

