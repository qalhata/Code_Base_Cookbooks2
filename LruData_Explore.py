# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 02:31:30 2018

@author: Shabaka
"""

# Import pandas
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# #########################################################
# #########################################################

# ######## Sample Project Format ########################

# Read in jobs file
test = pd.read_csv(r"C:\Users\Shabaka\ShabakaCodes\FV771_FaultFinder_a.csv", index_col=0)

# OR we could simply call up the folllowing in my case

# path = r'C:\Users\Shabaka\Documents\Eye_Tracking_Fixation-EDA'
# all_rec = glob.iglob(os.path.join(path, "*.csv"), recursive=True)
# dataframes = (pd.read_csv(f) for f in all_rec)
# test = pd.concat(dataframes, ignore_index=False)

# Review the first five lines of your DataFrame
print(test.head(5))
\
\
print('The following list illustrates the type of data in our Dataframe')
\
\
print('#################################################')
# Review the type of each column in your DataFrame
print(test.dtypes)
\
print('#################################################')

# %%
# How do we convert datestamp column?? (start_timestamp) to a datetime object

# test['start_timestamp'] = pd.to_datetime(test['start_timestamp'])

# Not sure about the line above. Still contemplating the representation
# of the data time stamp as datetime due to the scale
# of the capture time process in ms

# Set the datestamp columns as the index of the DataFrame

#%%
# Check the number of missing values in each column
print(test.isnull().sum())

#%%
# ###############
# Describe the time Series with some box plots

# Generate a boxplot
test.boxplot(fontsize=6, vert=False)
plt.show()

# Generate numerical summaries
print(test.describe())

#%%
# Print the name of the time series with the highest mean
print('T_Series wit highest Mean Value is start_frame_index')

# Print the name of the time series with the highest variability
print('T_Series with highest variability is ...',
      'depends on what this is for you')

#%%

# Plot all data in the time series - If you really need to #####

# A subset of the test DataFrame
test_subset = test[['duration', 'start_frame_index', 'confidence', 'dispersion']]
# test_subset = test[['duration', 'dispersion']]

# Print the first 5 rows of jobs_subset
print(test_subset.head())

# Create a facetted graph with 2 rows and 2 columns
ax = test_subset.plot(x=['duration'], y=['dispersion'],
                      subplots=True,
                      layout=(2,2),
                      sharex=False,
                      sharey=False,
                      linewidth=0.7,
                      fontsize=3,
                      legend=True)

plt.show()

#%%

# Plot time series dataset
ax = test.plot(linewidth=2, fontsize=5)

# Additional customizations
ax.set_xlabel('Process Time')
ax.legend(fontsize=5)

# Show plot
plt.show()

#%%

# One might want to plot an area chart instead
ax = test.plot.area(fontsize=5)

# Additional customizations
ax.set_xlabel('Process Time')
ax.legend(fontsize=5)

# Show plot
plt.show()

#%%

# It might be useful to plot time series dataset
# using the cubehelix color palette - a personal favourite

ax = test.plot(colormap='cubehelix', fontsize=5)

# Additional customizations
ax.set_xlabel('Date')
ax.legend(fontsize=6)

# Show plot
plt.show()

#%%
# Plot time series dataset using the PuOr color palette
ax = test.plot(colormap='PuOr', fontsize=5)

# Additional customizations
ax.set_xlabel('Date')
ax.legend(fontsize=6)

# Show plot
plt.show()

#%%
# ### One might need to document a Combined Summary Stats
# and Plot   - The follwoing FORMAT & SYNTAX should do the trick ###

# Plot the time series data in the DataFrame
ax = test.plot()

# Compute summary statistics of the df DataFrame
test_summary = test.describe()

# Add summary table information to the plot
ax.table(cellText=test_summary.values,
         colWidths=[0.3]*len(test.columns),
         rowLabels=test_summary.index,
         colLabels=test_summary.columns,
         loc='top')

#%%

# Say we were looking at a summary of only  the
# dataframe means

# Plot the test data
test_summ = test.describe()
ax = test_summ.plot(colormap='viridis', fontsize=6, linewidth=1)

# Add x-axis labels
ax.set_xlabel('Fixation_Processes', fontsize=6)

# Add summary table information to the plot
ax.table(cellText=test_summ.values,
         colWidths=[0.15]*len(test.columns),
         rowLabels=test_summ.index,
         colLabels=test_summ.columns,
         loc='top')

# Specify the fontsize and location of legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=6)

# Show plot
plt.show()

#%%

# A note on Subplots ####
\
# It is possible to create a "grid" of individual graphs by
# 'faceting' each time series by setting the subplots argument to True.
# The arguments that can be added are:

# layout: specifies the number of rows x columns to use.
# sharex and sharey: specifies whether the x-axis and y-axis
# values should be shared between plots.")

# Example

# We create a facetted graph with 2 rows and 4 columns
test.plot(subplots=True,
          layout=(4,3),
          sharex=False,
          sharey=False,
          colormap='viridis',
          fontsize=3,
          legend=False,
          linewidth=0.2)

plt.show()

# %%

# ####### Computing correlations between time series ###

# Compute the correlation between the "duration and dispersion" columns using the spearman method
print(test[['duration', 'dispersion']].corr(method='spearman'))

# Print the correlation between dur and dispersion columns
print("The spearman correlation value between the duration of fixations and the dispersion entity is ", 0.122)


#%%

# Compute the correlation between the multiple columns using the pearson method
print(test[['duration', 'dispersion', 'confidence']].corr(method='pearson'))

#%%

# Print the correlation between duration and dispersion columns
print("Pearson Correlation between duration and dispersion is", 0.16497)

# Print the correlation between "duration and confidence" columns
print("Pearson Correlation between duration and confidence is", 0.044183)

# Print the correlation between confidence and dispersion columns
print("Pearson Correlation between confidence and dispersion is",-0.172062)

#%%

# We can visualise correlation matrices to help us understand
# visually what the data is saying

# Example Code - FOrmat and SYntax #####

import seaborn  as sns

testdata_corr = test.corr()

sns.heatmap(testdata_corr)
plt.xticks(rotation=90)
plt.yticks(rotation=0)


#%%

# A more useful correlation matrix of the fixation DataFrame
test_dat = test.corr(method='spearman')


# Customize the heatmap of the corr_meat correlation matrix
sns.heatmap(test_dat,
            annot=True,
            linewidths=0.4,
            annot_kws={"size": 6})

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

#%%

# ### Seaborn Clustered Heatmaps ( i.e CLustermaps)

# Get correlation matrix of the fixation DataFrame
testdat_heat = test.corr(method='pearson')

# Again we customize the heatmap of the testdat_heat
# correlation matrix and rotate the x-axis labels
fig = sns.clustermap(testdat_heat,
                     row_cluster=True,
                     col_cluster=True,
                     figsize=(10, 10))

plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()

# #############################################################
# #############################################################
# #############################################################
#%%

# Extracting Frame Images

"""
import cv2
capture = cv2.VideoCapture("absolute_path_to_video/world.mp4")
status, img1 = capture.read() # extract the first frame
status, img2 = capture.read() # second frame...

"""

#%%