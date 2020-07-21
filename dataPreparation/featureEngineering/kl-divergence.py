import os
from math import log2
import numpy as np
from statistics import mean, stdev
import pandas as pd
from matplotlib import pyplot
import seaborn as sns
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
sns.set()


def listFiles(paths=None):
	files = []
	list_paths = []
	# r=root, d=directories, f = files
	# print(paths)
	for path in paths:
		for p, d, f in os.walk(path):
			list_paths.append(p)
			for file in f:
				if '.csv' in file:
					files.append(os.path.join(p, file))
	# files.append(file)
	return files


def pdist(x=None, bandwidth=None, plot=None):
	_min =int(x.min())
	_max = int(x.max())
	# x = x.to_numpy()
	# grid = GridSearchCV(KernelDensity(),
	#                     {'bandwidth': np.arange(0.1, 10.1, 0.1)},
	#                     cv=2)  # 20-fold cross-validation
	# grid.fit(x[:, None])
	# print(grid.best_params_)
	# bw = grid.best_params_
	bw = 0.15
	model = KernelDensity(bandwidth=bw, kernel='gaussian')
	featureX = x.reshape((len(x), 1))
	model.fit(featureX)
	# sample probabilities for a range of outcomes
	values = asarray([value for value in range(min, max)])
	values = values.reshape((len(values), 1))
	probabilities = model.score_samples(values)
	P = exp(probabilities)
	# print(P)
	# # plot the histogram and pdf
	if plot == 1:
		pyplot.hist(featureX, bins=50, density=True)
		pyplot.plot(values[:], P)
		pyplot.show()
	return P


def qdist(x=None, bandwidth=None, val1=None, val2=None, plot=None):
	# x = x.to_numpy()
	model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
	featureX = x.reshape((len(x), 1))
	model.fit(featureX)
	# sample probabilities for a range of outcomes
	values = asarray([value for value in range(val1, val2)])
	values = values.reshape((len(values), 1))
	probabilities = model.score_samples(values)
	Q = exp(probabilities)
	# print(Q)
	# # plot the histogram and pdf
	if plot == 1:
		pyplot.hist(featureX, bins=50, density=True)
		pyplot.plot(values[:], Q)
		pyplot.show()
	return Q


def outlierDetectionRemoval(data=None):
	# df = pd.read_csv(
	# 	r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features\audio.csv")
	# data = df[df.columns[10]]
	# # print(data)
	# print(data.shape)
	data_mean, data_std = mean(data), stdev(data)
	# identify outliers
	cut_off = data_std * 3
	lower, upper = data_mean - cut_off, data_mean + cut_off
	# identify outliers
	outliers = [x for x in data if x < lower or x > upper]
	# print(outliers)
	# print('Identified outliers: %d' % len(outliers))
	# remove outliers
	outliers_removed = [x for x in data if x >= lower and x <= upper]
	# print(type(outliers_removed))
	data_df = pd.Series(outliers_removed)
	# print('Non-outlier observations: %d' % len(outliers_removed))
	return data_df


def kl_divergence(result_path=None):
	files_features = listFiles(paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features'])
	files_labels = listFiles(paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\labels'])
	dfX = pd.read_csv(files_features[0])
	dfY = pd.read_csv(files_labels[0])
	_featureX = dfX[dfX.columns[1]]
	_featureY = dfY[dfY.columns[1]]
	featureX = outlierDetectionRemoval(_featureX).to_numpy()
	featureY = outlierDetectionRemoval(_featureY).to_numpy()
	min = featureX.min()
	max = featureX.max()

	print(min, max)
	# plt.hist(featureX, bins = 50)
	# plt.show()
	val1 =int(min)
	val2 = int(max)
	P = pdist(x=featureX, bandwidth=0, plot=1)
	Q = qdist(x=featureY, bandwidth=0.15, val1=val1, val2=val2, plot=1)
	sum = 0
	print('Size of featureY {}'.format(len(featureY)) )
	print('Size of Q  {}'.format(len(Q)))
	for i in range(len(Q)):
		# s = (P[i] * log2(P[i] / Q[i]))
		# print(s)
		# sum += s
		if P[i] > 0:
			if Q[i] > 0:
				s = (P[i] * log2(P[i]/Q[i]))
				# print(s)
				sum += s
	print(sum)
	# ks_stats[dfX.columns[col]] = [sum]
	# df_stats = pd.DataFrame(ks_stats)
	# df_stats.index = ['KL-div']
	# file_path = os.path.join(result_path, file_name)
	# df_stats.to_csv(file_path)

	# for i in range(0, len(files_features)):
	# 	file_name = os.path.basename(files_features[i])
	# 	print("Processing: ", file_name.replace('.csv', ''))
	# 	dfX = pd.read_csv(files_features[i])
	# 	dfY = pd.read_csv(files_labels[i])
	# 	ks_stats = {}
	#
	# 	for col in range(1, len(dfX.columns)):
	# 		_featureX = dfX[dfX.columns[col]]
	# 		_featureY = dfY[dfY.columns[col]]
	# 		featureX = outlierDetectionRemoval(_featureX)
	# 		featureY = outlierDetectionRemoval(_featureY)
	# 		P = pdist(x=featureX)
	# 		Q = qdist(x=featureY)
	# 		sum = 0
	# 		for i in range(len(Q)):
	# 			if P[i] > 0:
	# 				if Q[i] > 0:
	# 					s = (P[i] * log2(P[i]/Q[i]))
	# 					print(s)
	# 					sum += s
	# 		ks_stats[dfX.columns[col]] = [sum]
	# 	df_stats = pd.DataFrame(ks_stats)
	# 	df_stats.index = ['KL-div']
	# 	file_path = os.path.join(result_path, file_name)
	# 	# df_stats.to_csv(file_path)


def kde(result_path=None):
	files_features = listFiles(
		paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features'])
	files_labels = listFiles(
		paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\labels'])
	dfX = pd.read_csv(files_features[0])
	dfY = pd.read_csv(files_labels[0])
	_featureX = dfX[dfX.columns[1]]
	_featureY = dfY[dfY.columns[1]]
	featureX = outlierDetectionRemoval(_featureX).to_numpy()
	featureY = outlierDetectionRemoval(_featureY).to_numpy()
	grid = GridSearchCV(KernelDensity(),
	                    {'bandwidth': np.arange(0.1, 0.2, 0.1)},
	                    cv=2)  # 20-fold cross-validation
	grid.fit(featureX[:,None])
	print(grid.best_params_)
	bw = grid.best_params_
	print(grid.best_estimator_)
	k = grid.best_estimator_
	print(type(bw))
	print(bw['bandwidth'])
	values = asarray([value for value in range(30, 55)])
	P = np.exp(k.score_samples(values[:, None]))
	pyplot.hist(featureX, bins=50, density=True)
	pyplot.plot(values[:], P)
	pyplot.show()


	# model = KernelDensity(bandwidth=bw['bandwidth'], kernel='gaussian')
	# featureX = featureX.reshape((len(featureX), 1))
	# model.fit(featureX)
	# # sample probabilities for a range of outcomes
	# values = asarray([value for value in range(val1, val2)])
	# values = values.reshape((len(values), 1))
	# probabilities = model.score_samples(values)
	# P = exp(probabilities)
	# # print(P)
	# # # plot the histogram and pdf
	# if plot == 1:
	# 	pyplot.hist(featureX, bins=50, density=True)
	# 	pyplot.plot(values[:], P)
	# 	pyplot.show()


if __name__ == '__main__':
	# x = np.arange(-10, 10, 0.001)
	kl_divergence(result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results\results_kl')
	# kde(result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results\results_kl')