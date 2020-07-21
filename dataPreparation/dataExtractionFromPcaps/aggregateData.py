
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
	print("hello")
	list_combinedWithDNSNew = []
	for p, d, f in os.walk(dataset_path):
		if not 'backup_old_file' in p:
			if 'combined-with-dns-new' in p:
				list_combinedWithDNSNew.append(p)
	print(list_combinedWithDNSNew)
	for path in list_combinedWithDNSNew:
		print("Taking path - {}".format(path))
		for p, d, f in os.walk(path):
			p1 = path.split(os.sep)
			# print(p1)
			s = []
			for i in range(p1.index(dataset) + 1, len(p1) - 1):
				s.append(p1[i])
			# print(s)
			class_name = '_'.join(s)
			class_file_name = '_'.join(s) + '.csv'
			# print(class_file_name, 111)
			result_save_path = os.path.join(result_path, class_file_name)
			data4 = {'feature_file': [], 'pcap_file': [], 'label_file': [], 'birDir_flow_count': [], 'pcap_size_mb': []}
			df_class = pd.DataFrame(data4)
			for file in f:
				data_file_path = os.path.join(p, file)
				df = pd.read_csv(data_file_path, header=None, index_col=None)
				# print(data_file_path)
				# print(df.shape[0])
				df = df.drop(df[df.iloc[:, -1] == 'network_service'].index)
				# print(df.shape[0])
				count = df.shape[0]

				if dataset == 'Solana2020a':
					pcap_name = file.replace('_combined.csv', '.pcap')
					label_name = file.replace('_combined', '_label')
				else:
					pcap_name = file.replace('_combined.csv', '.pcap')
					label_name = file.replace('_combined', '_label')

				pcap_path = p.replace('combined-with-dns-new', 'pcaps')
				pcap = os.path.join(pcap_path, pcap_name)

				label_path = p.replace('combined-with-dns-new', 'with-dns-labels')
				label = os.path.join(label_path, label_name)

				try:
					pcap_size = os.path.getsize(pcap) / (1024 * 1024)
					pcap_size = round(pcap_size, 2)
				except:
					print('[error]: {} not found'.format(pcap))
					pcap_size = 0

				# =============================
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
	generateDataDetails(dataset='Solana2019a',
	                    dataset_path=r'C:\Users\Owner\mltat\data\mltat\Dataset\Solana2019a',
	                    result_path=r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\all_dataset',
	                    feat_train_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\splits\30-70percent\all-data_train",
	                    feat_test_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\splits\30-70percent\all-data_test",
	                    percentage=30
	                    )
