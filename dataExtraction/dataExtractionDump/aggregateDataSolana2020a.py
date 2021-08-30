import os
import pandas as pd
from dataAggregationSplit.splitTrainTest import splitTrainTest


def generateDataDetails(dataset=None,
                        dataset_path=None,
                        result_path=None,
                        feat_train_path=None,
                        feat_test_path=None,
                        percentage=None
                        ):
	list_features = []
	for p, d, f in os.walk(dataset_path):
		if 'features' in p:
			list_features.append(p)
	for path in list_features:
		print("Taking path - {}".format(path))
		for p, d, f in os.walk(path):
			p1 = path.split(os.sep)
			print(p1)
			s = []
			# print(p1.index(dataset))
			# print(len(p1))
			for i in range(p1.index(dataset) + 1, len(p1) - 1):
				s.append(p1[i])
			print(s)
			class_name = '_'.join(s)
			class_file_name = '_'.join(s) + '.csv'
			result_save_path = os.path.join(result_path, class_file_name)
			data4 = {'feature_file': [], 'pcap_file': [], 'label_file': [], 'birDir_flow_count': [], 'pcap_size_mb': []}
			df_class = pd.DataFrame(data4)
			for file in f:
				data_file_path = os.path.join(p, file)
				# print(file, 2222)
				df = pd.read_csv(data_file_path, header=None, index_col=None)
				df = df.drop(df[df.iloc[:, -1] == 'network_service'].index)
				count = df.shape[0]
				print(file, ' - ',  count)
				pcap_name = file.replace('_combined.csv', '.pcap')
				# pcap_name = file.replace('_feature.csv', '.pcap')
				label_name = file.replace('_combined', '_label')
				# label_name = file.replace('_feature', '_label')

				pcap_path = p.replace('features', 'pcaps')
				pcap = os.path.join(pcap_path, pcap_name)

				label_path = p.replace('features', 'labels')
				label = os.path.join(label_path, label_name)
				#
				try:
					pcap_size = os.path.getsize(pcap) / (1024 * 1024)
					pcap_size = round(pcap_size, 2)
				except:
					print('[error]: {} not found'.format(pcap))
					pcap_size = 0
				#
				# 			# =============================
				new_row4 = {'feature_file': data_file_path, 'pcap_file': pcap, 'label_file': label,
				            'birDir_flow_count': count, 'pcap_size_mb': pcap_size}
				df_class = df_class.append(new_row4, ignore_index=True)
			df_class['cumm_flow_count'] = df_class['birDir_flow_count'].cumsum()
			df_class['cumm_pcap_size_mb'] = df_class['pcap_size_mb'].cumsum()
			splitTrainTest(dataDetails=df_class,
			               _class=class_name,
			               feat_train_path=feat_train_path,
			               feat_test_path=feat_test_path,
			               percentage=percentage)
			df_class.to_csv(result_save_path)


if __name__ == '__main__':
	# ======= Dataset 2020a ===============
	# generateDataDetails(dataset='Solana2020a',
	#                     dataset_path=r'C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020a',
	#                     result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020\all_dataset",
	#                     feat_train_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020\splits\70-30percent\all-data_train",
	#                     feat_test_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020\splits\70-30percent\all-data_test",
	#                     percentage=70
	#                     )

	# generateDataDetails(dataset='Solana2020c',
	#                     dataset_path=r"C:\Users\Owner\mltat\data\mltat\Dataset\Solana2020c",
	#                     result_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\all_dataset\5class",
	#                     feat_train_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\splits\5Class\all-data_train",
	#                     feat_test_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\splits\5Class\all-data_test",
	#                     percentage=70
	#                     )
	generateDataDetails(dataset='Solana2020a',
	                    dataset_path=r"D:\Data\DataSets\Solana2020a",
	                    result_path=r"D:\Data\TestFiles\train_test_splits\Solana2020a\all_dataset",
	                    feat_train_path=r"D:\Data\TestFiles\train_test_splits\Solana2020a\splits\80-20percent\all-data_train",
	                    feat_test_path=r"D:\Data\TestFiles\train_test_splits\Solana2020a\splits\80-20percent\all-data_test",
	                    percentage=20
	                    )