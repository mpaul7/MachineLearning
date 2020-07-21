import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import os
from pathlib import Path


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


def test_labels(hdfsfilePath=None,
                labelFilePath=None,
                resultFilePath=None ):
	df_test = pd.read_csv(hdfsfilePath)
	df_labels = pd.read_csv(labelFilePath)
	dict_labels = df_labels.set_index('hash').T.to_dict('list')
	# print(dict_labels)
	df_test['hash'].replace(dict_labels, inplace=True)
	df_test.insert(36, 'actual_label', True)
	df_test.insert(37, 'pred_label', '')
	df_test['actual_label'] = df_test['hash']
	df_test.drop(['# sip', 'sport', 'dip', 'dport', 'hash', 'ml_cat'], axis=1, inplace=True)
	df_test.to_csv(resultFilePath, index=False)


if __name__ == '__main__':
	# files = listFiles(paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test_v2\Solana2019\splits\70-30percent\7Class\test_30per'])
	# print(files)
	test_labels(hdfsfilePath=r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\merged_hdfs_task-35i.csv",
                labelFilePath=r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\merged_labels_Solana2020a_30per.csv",
                resultFilePath=r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\merged_hdfs_with_labels_task-35i.csv")