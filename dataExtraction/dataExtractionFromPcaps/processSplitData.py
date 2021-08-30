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
	# print(files)
	list_df = []
	for f in files:
		print('Processing : {}'.format(f))
		df1 = pd.read_csv(f)
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
	counter = 1
	for f in files:
		print("Processing file : {}".format(f))
		df = pd.read_csv(f)
		col = df.columns[0]
		list_pcaps = df[col].tolist()
		PCAP_file_name = result_path + '\Test-pcap_Solana2019_50per-' + str(counter) + '.pcap'
		print(PCAP_file_name)
		merge_pcap_file(list_pcaps, PCAP_file_name)
		counter += 1


def merge_pcap_file(pcap_fils, output):
	command = ["mergecap", "-F", "pcap", "-w", output]
	for f in pcap_fils:
		command.append(f)
	print('Merging pcap file ...' ) #  format(f))
	sb.call(command, )


def merge_pcap_file_dl(keyname, tdlist):
	index = 0
	for sublist in tdlist:
		output_file = "{}_{}_test_pcap.pcap".format(keyname, index)
		merge_pcap_file(sublist, output_file)
		index += 1


if __name__ == '__main__':

	# ======= Process Features ===========================
	df = getFeatures(path=[r"E:\Data\TestFiles\train_test_splits\Solana2020d\splits\50-50percent\6Class\train_50per"],
	                 result_path=r'E:\Data\TestFiles\train_test_splits\Solana2020d\merged_files\Solana2020d-50-50\6Class_app_label\merged_train_data_Solana2020d_ 50per.csv')

	mergeFeatures(df=df,
	              result_path=r'E:\Data\TestFiles\train_test_splits\Solana2020d\final_test_data\Solana2020d-50-50\6Class_app_label\train_data_6class_Solana2020d_50per.csv')

	# =======Process labels ==============================
	# df = getLabels(
	# 	path=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\splits\50-50percent\7Class\test_50per'],
	# 	result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\merged_files\Solana2019-50-50\merged_labels_Solana2019a_50per.csv')
	#
	# mergeLables(df=df,
	#             result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\final_test_data\Solana2019-50-50\merged_labels_Solana2019a_50per.csv')

	# ======= Process Pcaps ===============================
	# splitPcaps(path=[r"D:\Data\TestFiles\train_test_splits\Solana2020a\splits\80-20percent\6Class\train_80per"],
	#            result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\merged_files\Solana2019-70-30\pcaps',
	#            pcap_size_mb=380)

	# getPcaps(path=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\merged_files\Solana2019-70-30\pcaps'],
	#          result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\final_test_data\Solana2019-70-30\pcaps')