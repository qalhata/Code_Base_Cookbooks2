# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:46:57 2017

@author: Shabaka
"""
import numpy as np
import matplotlib.pyplot as plt

from Bernoulli_Trial import perform_bernoulli_trials
from ecdf_func import ecdf

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults - two arguments: the number of trials n - in
# this case 100 - and the probability of success p - in this case the
# probability of a default, which is 0.05.
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(10000, 0.05)


# Plot the histogram with default number of bins;
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

# Compute ECDF: x, y

x, y = ecdf(n_defaults)

# Plot the ECDF with labeled axes

_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.002)
plt.xlabel('n_defaults')
plt.ylabel('ECDF')


# Show the plot
plt.show()

# ########################################### #

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(10000)

# Compute the number of defaults
for i in range(10000):
    n_defaults[i] = perform_bernoulli_trials(1000, 0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

# Compute bin edges: bins
bins = np.arange(-0.5, max(n_defaults + 1.5) - 0.5)

# Generate histogram
_ = plt.hist(n_defaults, normed=True, bins=bins)

# Set margins
plt.margins(0.02)

# Label axes
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('Binomial PMF')

# Compute the number of 100-loan simulations
# with 10 or more defaults: n_lose_money

n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
# print('Probability of losing money =', n_lose_money / len(n_defaults)

# Take 10,000 samples out of the binomial distribution: n_defaults


# Compute CDF: x, y
x, y = ecdf(n_defaults)

# Plot the CDF with axis labels
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.002)
plt.xlabel('Defaults out of 100 loans')
plt.ylabel('ECDF')

# Show the plot
plt.show()

# ############################## #

# Compute bin edges: bins
bins = np.arange(-0.5, max(n_defaults + 1.5) - 0.5)

# Generate histogram
_ = plt.hist(n_defaults, normed=True, bins=bins)

# Set margins
plt.margins(0.02)

# Label axes
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('Binomial PMF')


# Show the plot
plt.show()


# ################################## #

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))

# ############################################## #