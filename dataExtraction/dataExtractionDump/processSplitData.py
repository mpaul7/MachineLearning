import os
import pandas as pd
import shutil
import subprocess as sb


def listFiles(paths=None):
	print("hello")
	files = []
	list_paths = []
	# r=root, d=directories, f = files
	print(paths)
	for path in paths:
		for p, d, f in os.walk(path):
			list_paths.append(p)
			for file in f:
				if '.csv' in file:
					files.append(os.path.join(p, file))
	# files.append(file)
	return files


def getFeatures(path=None, result_path=None):
	files = listFiles(paths=path)
	print(files)
	list_df = []
	for f in files:
		print('Processing : {}'.format(f))
		df1 = pd.read_csv(f)
		print(df1)
		list_df.append(df1)
	df = pd.concat(list_df)
	df.drop(['pcap_file', 'label_file', 'birDir_flow_count', 'pcap_size_mb', 'cumm_flow_count', 'cumm_pcap_size_mb'],
	        inplace=True, axis=1)
	df.to_csv(result_path, index=False)
	print('Total Feature files - {}'.format(df.shape[0]))
	return df


def mergeFeatures(df=None, result_path=None):
	list_df = []
	for index, rows in df.iterrows():
		col = df.columns[0]
		df1 = pd.read_csv(rows[col], header=None)
		list_df.append(df1)
	df2 = pd.concat(list_df)
	print('Total_flows {}'.format(df2.shape[0]))
	df2.to_csv(result_path, header=None, index=False)


def getLabels(path=None, result_path=None):
	files = listFiles(paths=path)
	# print(files)
	list_df = []
	for f in files:
		print('Processing file : {}'.format(f))
		df1 = pd.read_csv(f)
		list_df.append(df1)
	df = pd.concat(list_df)
	print('total Label files {}'.format(df.shape[0]))
	df.drop(['feature_file', 'pcap_file', 'birDir_flow_count', 'pcap_size_mb', 'cumm_flow_count', 'cumm_pcap_size_mb'],
	        inplace=True, axis=1)
	df.to_csv(result_path, index=False)
	return df


def mergeLables(df=None, result_path=None):
	list_df = []
	list_filesNotFound = []
	for index, rows in df.iterrows():
		try:
			df1 = pd.read_csv(rows['label_file'])
			list_df.append(df1)
		except:
			list_filesNotFound.append(rows['label_file'])
	print("List of files not found {}".format(len(list_filesNotFound)))
	print(list_filesNotFound)
	df2 = pd.concat(list_df)
	print('Total flows {}'.format(df2.shape[0]))
	df2.to_csv(result_path, index=False)


def splitPcaps(path=None, result_path=None, pcap_size_mb=None):
	files = listFiles(paths=path)
	list_df = []
	counter = 1
	for f in files:
		df1 = pd.read_csv(f)
		list_df.append(df1)
	df = pd.concat(list_df)
	df['cumm_pcap_size_mb'] = df['pcap_size_mb'].cumsum()
	df.reset_index(inplace=True)
	all_train_pcap_path = result_path + '\All-train_pcaps_7Class__Solana2019a_30per.csv'
	df.drop(df[df['pcap_size_mb'] > pcap_size_mb].index, inplace=True)
	df.drop(['index', 'cumm_flow_count', 'cumm_pcap_size_mb'], axis=1, inplace=True)
	df.to_csv(all_train_pcap_path, index=False)
	while df.shape[0] > 0:
		PCAP_file_name = 'test-pcap' + str(counter) + '.csv'
		trainPCAP_path = os.path.join(result_path, PCAP_file_name)
		print('Processing file : {}'.format(trainPCAP_path))
		df['cumm_pcap_size_mb'] = df['pcap_size_mb'].cumsum()
		print(df.shape)
		df_filtered = df[df['cumm_pcap_size_mb'] < pcap_size_mb]
		print(df_filtered.shape)
		df.drop(df[df['cumm_pcap_size_mb'] < pcap_size_mb].index, inplace=True)
		counter += 1
		df_filtered.drop(['feature_file', 'label_file', 'birDir_flow_count', 'pcap_size_mb', 'cumm_pcap_size_mb'],
		                 inplace=True, axis=1)
		df_filtered.to_csv(trainPCAP_path, index=False)


def getPcaps(path=None, result_path=None):
	files = listFiles(paths=path)
	print(files)
	counter = 1
	# ============================
	# PCAP_file_name = result_path + '\merged-video-stream-youtube2' + '.pcap'
	# print(PCAP_file_name)
	# merge_pcap_file(files, PCAP_file_name)
	# ==================================
	for f in files:
		print("Processing file : {}".format(f))
		df = pd.read_csv(f)
		col = df.columns[0]
		list_pcaps = df[col].tolist()
		PCAP_file_name = os.path.join(result_path + r'\6Class_test_pcap_Solana2020d_50per-{}.pcap'.format(str(counter)))
		print(PCAP_file_name)
		merge_pcap_file(list_pcaps, PCAP_file_name)
		counter += 1


def merge_pcap_file(pcap_fils, output):
	command = ["mergecap", "-F", "pcap", "-w", output]
	for f in pcap_fils:
		print(f)
		command.append(f)
	print('Merging pcap file ...' )
	sb.call(command)


def merge_pcap_file_dl(keyname, tdlist):
	index = 0
	for sublist in tdlist:
		output_file = "{}_{}_test_pcap.pcap".format(keyname, index)
		merge_pcap_file(sublist, output_file)
		index += 1


def getFlowsLabelWise(path=None, result_path=None):
	files = listFiles(paths=path)
	print(files)
	list_df = []
	for f in files:
		print('Processing : {}'.format(f))
		df1 = pd.read_csv(f, header=None)
		list_df.append(df1)
		# print(df1)
	df = pd.concat(list_df)
	# print(df)
	# print(df.columns)
	print(df.groupby(df.columns[-1]).count())
	# df.drop(['pcap_file', 'label_file', 'birDir_flow_count', 'pcap_size_mb', 'cumm_flow_count', 'cumm_pcap_size_mb'], inplace=True, axis=1)
	df.to_csv(result_path)
	print('Total Feature files - {}'.format(df.shape[0]))
	return df


def get_pcap_twc(file=None, result_path=None):
	# files = listFiles(paths=path)
	df = pd.read_csv(file)
	pcap_files = df['pcap_file'].to_list()
	print(pcap_files)
	command = ['cp', '', '']
	for file in pcap_files:
		print(file)
		command[-2] = file
		# command[-1] = \
		target = result_path
		print(command)
		shutil.copy(file, target)
		# sb.call(command)


if __name__ == '__main__':
	print("hello")
	# ======= Process Features =============================
	df = getFeatures(path=[r"D:\Data\TestFiles\train_test_splits\Solana2020c\splits\50-50percent\3class\test-50per"],
	                 result_path=r"D:\Data\TestFiles\train_test_splits\Solana2020c\merged_files\Solana2020c-50-50\3Class\3Class_merged_features_Solana2020c_50per.csv")
	# print(df)

	mergeFeatures(df=df,
	              result_path=r"D:\Data\TestFiles\train_test_splits\Solana2020c\final_test_data\Solana2020c-50-50\3Class\3Class_test_features_Solana2020c_50per.csv")

	# # =======Process labels ==============================
	# df = getLabels(
	# 	path=[r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana-5G\splits\50-50percent\6Class\test_50per"],
	# 	result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana-5G\merged_files\Solana2020d-50-50\6Class_merged_label_Solana2020d_50per.csv")
	#
	# mergeLables(df=df,
	#             result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana-5G\final_test_data\Solana2020d-50-50\6Class\6Class_labels_Solana2020d_50per.csv")

	# ======= Process Pcaps ===============================
	# splitPcaps(path=[r"D:\Data\TestFiles\train_test_splits\Solana2020a\splits\80-20percent\6Class-2\test-20per"],
	#            result_path=r"C:\Users\Owner\projects3\twc\cpkt\cpkt\twc\twc\data\task-7a-pcap-based\merged_data\test",
	#            pcap_size_mb=350)

	# getPcaps(path=[r"C:\Users\Owner\projects3\twc\cpkt\cpkt\twc\twc\data\task-7a-pcap-based\merged_data\test"],
	#          result_path=r"C:\Users\Owner\projects3\twc\cpkt\cpkt\twc\twc\data\task-7a-pcap-based\final_data\test_pcaps")
	# get_pcap_twc(file = r"D:\Data\TestFiles\train_test_splits\Solana2019\splits\80-20percent\6Class\test_20per\video-stream_youtube-20.csv",
	#              result_path=r"C:\Users\Owner\projects3\twc\cpkt\cpkt\twc\twc\data\task-7b-pcap-based\test-pcaps\video_stream\youtube")
	# ================pa===========================================
	# getFlowsLabelWise(path=[r"C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\video-stream\youtube\features"],
	#                   result_path=r'C:\Users\Owner\mltat\mltat\test_data\Qosmos\data\audio-stream\spotify\merged_features_deezer.csv')

