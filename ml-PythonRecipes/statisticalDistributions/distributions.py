

import numpy as np
from matplotlib import pyplot
from numpy.random import normal
from numpy import mean
from numpy import std
from scipy.stats import norm
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity

def test():

	# example of parametric probability density estimation


	# generate a sample
	sample = normal(loc=50, scale=5, size=1000)
	# calculate parameters
	sample_mean = mean(sample)
	sample_std = std(sample)
	print(sample)
	print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
	# define the distribution
	dist = norm(sample_mean, sample_std)
	# sample probabilities for a range of outcomes
	values = [value for value in range(30, 70)]
	probabilities = [dist.pdf(value) for value in values]
	# plot the histogram and pdf
	pyplot.hist(sample, bins=10, density=True)
	pyplot.plot(values, probabilities)
	pyplot.show()

def test2():
	# generate a sample
	sample1 = normal(loc=20, scale=1, size=300)
	sample2 = normal(loc=40, scale=1, size=700)
	sample = hstack((sample1, sample2))
	# fit density
	model = KernelDensity(bandwidth=0.2, kernel='gaussian')
	sample = sample.reshape((len(sample), 1))
	model.fit(sample)
	# sample probabilities for a range of outcomes
	values = asarray([value for value in range(1, 60)])
	print(type(values))
	values = values.reshape((len(values), 1))
	probabilities = model.score_samples(values)
	probabilities = exp(probabilities)
	# plot the histogram and pdf
	pyplot.hist(sample, bins=50, density=True)
	pyplot.plot(values[:], probabilities)
	pyplot.show()


if __name__ == '__main__':
	# test()
	test2()