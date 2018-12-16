# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:58:28 2018

@author: Shabaka
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import weibull


test = pd.read_csv(r"C:\Users\Shabaka\ShabakaCodes\FV771_FaultFinder_a.csv", index_col=0)

# Review the first five lines of your DataFrame
print(test.head(5))
\
#%%
designer = weibull.Design(target_cycles=10000, reliability=0.9,
                          confidence_level=0.90, expected_beta=1.5)

# print(f'Minimum number of units for 10000: {designer.num_of_units(test_cycles=10000)}')

# print(f'Minimum hours for 20 units: {designer.num_of_cycles(num_of_units=20)}')

analysis = weibull.Analysis(fail_times, suspensions, unit='hour')
analysis.fit(method='mle', confidence_level=0.6)
print(analysis.stats)
analysis.probplot(file_name='gallery-probplot.png')
analysis.pdf(file_name='gallery-pdf.png')
analysis.hazard(file_name='gallery-hazard.png')
analysis.sf(file_name='gallery-survival.png')
analysis.fr(file_name='gallery-fr.png')