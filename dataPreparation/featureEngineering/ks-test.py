import os
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from statistics import mean, stdev
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.distributions.empirical_distribution import ECDF


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

	# data_df.plot(kind='box', subplots=True, sharex=False, sharey=False)
	# plt.xlabel('Value')
	# plt.ylabel('Count')
	# plt.title('audio-stream - ' + df.columns[1])
	# plt.show()
	return data_df


def plot(x=None):
	df = pd.read_csv(r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features\audio.csv")
	x = df[df.columns[10]]
	# x.hist()
	x.plot(kind='box', subplots=True, sharex=False, sharey=False)
	plt.xlabel('Value')
	plt.ylabel('Count')
	plt.title('audio-stream - ' + df.columns[1])
	plt.show()


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


def ksTest_rawValues(result_path=None):
	files_features = listFiles(
		paths=[r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features"])
	files_labels = listFiles(
		paths=[r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels"])
	ks_stats_per_feature_allclass = {}
	df_stats_all_class = pd.DataFrame()
	for i in range(0, len(files_features)):
		file_name = os.path.basename(files_features[i])
		print("Processing: ", file_name.replace('.csv', ''))
		dfX = pd.read_csv(files_features[i])
		dfY = pd.read_csv(files_labels[i])
		ks_stats_per_feature_perclass = {}

		list = []
		list_distance = []

		for col in range(0, len(dfX.columns)):
			_featureX = dfX[dfX.columns[col]]
			_featureY = dfY[dfY.columns[col]]

			featureX = outlierDetectionRemoval(_featureX)
			featureY = outlierDetectionRemoval(_featureY)

			# ===== Ks - test ------
			res_ks = ks_2samp(featureX, featureY)
			list_distance.append(res_ks[0])
			Ho_ks = 'accepted'
			if res_ks[1] <= 0.05:
				Ho_ks = 'rejected'
				list.append(dfX.columns[col])
			# ks_stats_per_feature_perclass[dfX.columns[col]] = [res_ks[0], res_ks[1], Ho_ks]
			ks_stats_per_feature_perclass[dfX.columns[col]] = [res_ks[0]]
			# ks_stats_per_feature_allclass[dfX.columns[col]] = [res_ks[0]]
		# print(ks_stats_per_feature_allclass)
		# df_stats_all_class.append(ks_stats_per_feature_allclass)
		df_stats_per_class = pd.DataFrame(ks_stats_per_feature_perclass)
		# df_stats_per_class.index = ['statistics', 'p-value', 'Ho_ks']
		df_stats_per_class.index = ['statistics']
		print(df_stats_all_class)
		file_path = os.path.join(result_path, file_name)
		df_stats_per_class.to_csv(file_path)

	print("Process finished...")


def ksCrossTest(result_path=None):
	print("hello")
	_dict1 = {1: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio_chat.csv"],
	          2: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\network_service.csv"],
	          3: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\file_transfer.csv"],
	          4: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\video.csv"],
	          5: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\p2p.csv"],
	          6: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\social_media.csv"],
	          7: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio_chat.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\network_service.csv"],
	          8: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio_chat.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\file_transfer.csv"],
	          9: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio_chat.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\video.csv"],
	          10: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio_chat.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\p2p.csv"],
	          11: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio_chat.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\social_media.csv"],
	          12: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\file_transfer.csv"],
	          13: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\video.csv"],
	          14: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\p2p.csv"],
	          15: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\social_media.csv"],
	          16: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\video.csv"],
	          17: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\p2p.csv"],
	          18: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\social_media.csv"],
	          19: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\p2p.csv"],
	          20: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\social_media.csv"],
	          21: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\social_media.csv"],
	          22: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\audio_chat.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio.csv"],
	          23: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio.csv"],
	          24: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio_chat.csv"],
	          25: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio.csv"],
	          26: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio_chat.csv"],
	          27: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\network_service.csv"],
	          28: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio.csv"],
	          29: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio_chat.csv"],
	          30: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\network_service.csv"],
	          31: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\file_transfer.csv"],
	          32: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio.csv"],
	          33: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio_chat.csv"],
	          34: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\network_service.csv"],
	          35: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\file_transfer.csv"],
	          36: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\video.csv"],
	          37: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio.csv"],
	          38: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\audio_chat.csv"],
	          39: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\network_service.csv"],
	          40: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\file_transfer.csv"],
	          41: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\video.csv"],
	          42: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\labels\p2p.csv"]}

	# =======================================================================================
	for k, v in _dict1.items():
		pathX = _dict1[k][0]
		pathY = _dict1[k][1]
		no = k
		dfX = pd.read_csv(pathX)
		dfY = pd.read_csv(pathY)
		file_name_X = os.path.basename(pathX).replace('.csv', '')
		file_name_Y = os.path.basename(pathY).replace('.csv', '')
		print(file_name_X)
		print(file_name_Y)

		file_name = str(no) + '_' + file_name_X + '-' + file_name_Y
		print(file_name)
		list_distance = []
		ks_stats = {}
		for col in range(0, len(dfX.columns)):
			_featureX = dfX[dfX.columns[col]]
			_featureY = dfY[dfY.columns[col]]

			featureX = outlierDetectionRemoval(_featureX)
			featureY = outlierDetectionRemoval(_featureY)
			# ===== Ks - test ------
			res_ks = ks_2samp(featureX, featureY)
			list_distance.append(res_ks[0])
			Ho_ks = 'same'
			if res_ks[1] <= 0.05:
				Ho_ks = 'different'
			# ks_stats[dfX.columns[col]] = [res_ks[0], res_ks[1], Ho_ks]
			ks_stats[dfX.columns[col]] = [res_ks[0]]
		df_stats = pd.DataFrame(ks_stats)
		print(df_stats)
		# df_stats.index = ['statistics', 'p-value', 'dist_ks']
		df_stats.index = [no]
		_file_name = file_name + '.csv'
		file_path = os.path.join(result_path, _file_name)
		df_stats.to_csv(file_path)



def edf():
	dfX= pd.read_csv(r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features\p2p.csv")
	dfY = pd.read_csv(r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\labels\p2p.csv")

	_featureX = dfX[dfX.columns[16]]
	_featureY = dfY[dfY.columns[16]]

	featureX = outlierDetectionRemoval(_featureX)
	featureY = outlierDetectionRemoval(_featureY)

	plt1 = pyplot
	plt2 = pyplot

	# plt1.hist(featureX, bins=50, alpha=0.5, label='features')
	# plt1.hist(featureY, bins=50, alpha=0.5, label='labels')
	# plt1.hist([featureX, featureY], bins=50, alpha=0.5, label=['features', 'labels'])
	# plt1.xlabel('X')
	# plt1.ylabel('Count')
	# plt1.title('p2p- ' + 'R12')
	# plt1.legend(loc='upper right')
	# plt1.savefig(r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results\results_ks-test_ecdf\hist-R12.png')
	res_ks = ks_2samp(featureX, featureY)
	print(res_ks)
	ecdfX = ECDF(featureX)
	ecdfY = ECDF(featureY)
	plt2.plot(ecdfX.x, ecdfX.y)
	plt2.plot(ecdfY.x, ecdfY.y)
	plt2.xlabel('X')
	plt2.ylabel('Cummulative Probability')
	plt2.title('p2p - ' + 'R12')
	plt2.savefig(r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results\results_ks-test_ecdf\ecdf-R12.png')
	# pyplot.show()

def test():
	# _dict = {1: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30\ks-test\features\audio.csv", r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30\ks-test\labels\audio_chat.csv"]}
	# print(_dict)
	# print(_dict[1][0])
	# print(_dict[1][1])

	_dict1 = {1: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio_chat.csv"],
	          2: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\network_service.csv"],
	          3: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\file_transfer.csv"],
	          4: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\video.csv"],
	          5: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\p2p.csv"],
	          6: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\social_media.csv"],
	          7: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio_chat.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\network_service.csv"],
	          8: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio_chat.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\file_transfer.csv"],
	          9: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio_chat.csv",
	              r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\video.csv"],
	          10: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio_chat.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\p2p.csv"],
	          11: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio_chat.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\social_media.csv"],
	          12: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\file_transfer.csv"],
	          13: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\video.csv"],
	          14: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\p2p.csv"],
	          15: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\social_media.csv"],
	          16: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\video.csv"],
	          17: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\p2p.csv"],
	          18: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\social_media.csv"],
	          19: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\p2p.csv"],
	          20: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\social_media.csv"],
	          21: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\social_media.csv"]}

	_dict2 = {22: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\audio_chat.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio.csv"],
	          23: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio.csv"],
	          24: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\network_service.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio_chat.csv"],
	          25: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio.csv"],
	          26: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio_chat.csv"],
	          27: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\file_transfer.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\network_service.csv"],
	          28: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio.csv"],
	          29: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio_chat.csv"],
	          30: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\network_service.csv"],
	          31: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\video.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\file_transfer.csv"],
	          32: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio.csv"],
	          33: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio_chat.csv"],
	          34: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\network_service.csv"],
	          35: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\file_transfer.csv"],
	          36: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\p2p.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\video.csv"],
	          37: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio.csv"],
	          38: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\audio_chat.csv"],
	          39: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\network_service.csv"],
	          40: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\file_transfer.csv"],
	          41: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\video.csv"],
	          42: [r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\features-2019\social_media.csv",
	               r"C:\Users\Owner\mltat\data\mltat\TestFiles\task30-b\labels-2019\p2p.csv"]}

	for k, v in _dict1.items():
		print(k)

	for k, v in _dict2.items():
		print(k, v)


if __name__ == '__main__':
	# edf()

	# ====== Plots ======
	# densityPlot()

	# ===== ks Test =====
	# ksTest_zscore(result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results\results_ks-test_dValue_eachClass")
	# ksTest_rawValues(result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\results-ks-test\ks-test-univariate-intra-class")
	# ksTest(result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\results")

	# ====== ks-cross-test ====
	ksCrossTest(result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\results-ks-test\ks-test-univariate-cross-class")
