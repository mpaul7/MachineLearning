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
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott, select_bandwidth
from datetime import datetime
from scipy.special import kl_div
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
	P = 0
	low = int(x.min())
	high = min([int(x.max()), 1000000])
	# grid = GridSearchCV(KernelDensity(),
	#                     {'bandwidth': np.arange(0.1, 12, 0.2)},
	#                     cv=2)  # 20-fold cross-validation
	# grid.fit(x[:, None])
	# # print(grid.best_params_)
	# bw = grid.best_params_
	# print('P : {}'.format(grid.best_estimator_))
	# k = grid.best_estimator_
	# # print(type(bw))
	# # print(bw['bandwidth'])
	# values = asarray([value for value in range(low, high)])
	# try:
	# 	P = np.exp(k.score_samples(values[:, None]))
	# except :
	# 	P=0
	# bw = bw_silverman(x)
	bw = bw_scott(x)
	print('P- bw - {}'.format(bw))
	# =========================================
	# bw = 0.15
	try:
		model = KernelDensity(bandwidth=bw, kernel='gaussian')
	except:
		return P
	featureX = x.reshape((len(x), 1))
	model.fit(featureX)
	# sample probabilities for a range of outcomes
	values = asarray([value for value in range(low, high)])
	values = values.reshape((len(values), 1))

	# =========================================
	try:
		probabilities = model.score_samples(values)
		P = exp(probabilities)
	except :
		P=0
	if plot == 1:
		pyplot.hist(x, bins=50, density=True)
		pyplot.plot(values[:], P)
		pyplot.show()
	return P


def qdist(x=None, bandwidth=None, plot=None):
	Q=0
	low = int(x.min())
	high = min([int(x.max()), 1000000])
	# grid = GridSearchCV(KernelDensity(),
	#                     {'bandwidth': np.arange(0.1, 12, 0.2)},
	#                     cv=2)  # 20-fold cross-validation
	# grid.fit(x[:, None])
	# # print(grid.best_params_)
	# bw = grid.best_params_
	# print('P : {}'.format(grid.best_estimator_))
	# k = grid.best_estimator_
	# # print(type(bw))
	# # print(bw['bandwidth'])
	# values = asarray([value for value in range(low, high)])
	# try:
	# 	P = np.exp(k.score_samples(values[:, None]))
	# except :
	# 	P=0
	# bw = bw_silverman(x)
	bw = bw_scott(x)
	print('Q- bw - {}'.format(bw))
	# =========================================
	# bw = 0.15
	try:
		model = KernelDensity(bandwidth=bw, kernel='gaussian')
	except:
		return Q
	featureX = x.reshape((len(x), 1))
	model.fit(featureX)
	# sample probabilities for a range of outcomes
	values = asarray([value for value in range(low, high)])
	values = values.reshape((len(values), 1))

	# =========================================
	try:
		probabilities = model.score_samples(values)
		Q = exp(probabilities)
	except:
		Q = 0
	if plot == 1:
		pyplot.hist(x, bins=50, density=True)
		pyplot.plot(values[:], Q)
		pyplot.show()
	return Q


def outlierDetectionRemoval(data=None):
	data_mean, data_std = mean(data), stdev(data)
	# identify outliers
	cut_off = data_std * 3
	lower, upper = data_mean - cut_off, data_mean + cut_off
	# identify outliers
	outliers = [x for x in data if x < lower or x > upper]
	# remove outliers
	outliers_removed = [x for x in data if x >= lower and x <= upper]
	data_df = pd.Series(outliers_removed)
	# print('Non-outlier observations: %d' % len(outliers_removed))
	return data_df


def kl_divergence(result_path=None):
	files_features = listFiles(paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\Solana2019a\features'])
	files_labels = listFiles(paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\Solana2019a\labels'])
	for i in range(0, len(files_features)):
		file_name = os.path.basename(files_features[i])
		print(file_name.replace('.csv', ''))
		if file_name.replace('.csv', '') == 'network_service':
			continue
		dfX = pd.read_csv(files_features[i])
		dfY = pd.read_csv(files_labels[i])
		ks_stats = {}

		for col in range(1, len(dfX.columns)):
			print("Processing: {} - {}".format(file_name.replace('.csv', ''), dfX.columns[col]), )
			if dfX.columns[col] == 'prot':
				continue
			_featureX = dfX[dfX.columns[col]]
			_featureY = dfY[dfY.columns[col]]
			featureX = outlierDetectionRemoval(_featureX).to_numpy()
			featureY = outlierDetectionRemoval(_featureY).to_numpy()
			count = max([featureX.max(), featureY.max()])
			print(count)
			# if count > 100000:
			# 	continue
			P = pdist(x=featureX, plot=0)
			Q = qdist(x=featureY, plot=0)
			print('P - {}, Q - {}'.format(len(P), len(Q)))
			print(P)
			print(Q)
			# ====== new method ======================
			res = kl_div(P, Q)
			print(res)
			ks_stats[dfX.columns[col]] = [res]
			# ====== Old Method ======================
			# sum = 0
			# try:
			# 	itr = min([len(P), len(Q)])
			# except:
			# 	continue
			# print('P - {}, Q - {}, itr -{}'.format(len(P), len(Q), itr))
			# if itr == 0:
			# 	continue
			# for i in range(1, itr):
			# 	if P[i] > 0:
			# 		if Q[i] > 0:
			# 			s = (P[i] * log2(P[i]/Q[i]))
			# 			sum += s
			# ks_stats[dfX.columns[col]] = [sum]
			# ==========================================
		df_stats = pd.DataFrame(ks_stats)
		df_stats.index = ['KL-div']
		file_path = os.path.join(result_path, file_name)
		print(file_path)
		df_stats.to_csv(file_path)


if __name__ == '__main__':
	now = datetime.now()
	print("current time = ", now.strftime("%H:%M:%S"))
	kl_divergence(result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results\results_kl\results_kl_1000000_scott_bw')
	# kde(result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results\results_kl')
	print("current time = ", now.strftime("%H:%M:%S"))

# dfX = pd.read_csv(files_features[0])
# dfY = pd.read_csv(files_labels[0])
# _featureX = dfX[dfX.columns[1]]
# _featureY = dfY[dfY.columns[1]]
# featureX = outlierDetectionRemoval(_featureX).to_numpy()
# featureY = outlierDetectionRemoval(_featureY).to_numpy()
# min = featureX.min()
# max = featureY.max()
# P = pdist(x=featureX, plot=1)
# Q = qdist(x=featureY, plot=1)
# sum = 0
# print('Size of featureY {}'.format(len(featureY)) )
# print('Size of Q  {}'.format(len(Q)))
# for i in range(len(Q)):
# 	# s = (P[i] * log2(P[i] / Q[i]))
# 	# print(s)
# 	# sum += s
# 	if P[i] > 0:
# 		if Q[i] > 0:
# 			s = (P[i] * log2(P[i]/Q[i]))
# 			# print(s)
# 			sum += s
# print(sum)
# ks_stats[dfX.columns[col]] = [sum]
# df_stats = pd.DataFrame(ks_stats)
# df_stats.index = ['KL-div']
# file_path = os.path.join(result_path, file_name)
# df_stats.to_csv(file_path)