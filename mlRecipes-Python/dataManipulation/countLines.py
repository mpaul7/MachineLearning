import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import csv
import os
import pandas as pd

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
	# return list_paths, files
	return files


def countLines(listFiles=None, result_path=None):
	data4 = {'file_name': [], 'flow_count': []}
	df_stats = pd.DataFrame(data4)
	for f in listFiles:
		print()
		file = open(f)
		reader = csv.reader(file)
		lines = len(list(reader))
		file_name = os.path.basename(f)
		print("file: {}, lines - {}".format(f, (lines - 1)))
		new_row4 = {'file_name': file_name, 'flow_count': lines-1}
		df_stats = df_stats.append(new_row4, ignore_index=True)

	if result_path != None:
		df_stats.to_csv(result_path, index=False)


if __name__ == '__main__':
	listFiles = listFiles(paths=[r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\splits\20-80percent\all-data_train"])
	countLines(listFiles=listFiles,
	           result_path=None) #r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2019\splits\50-50percent\flow_count\flow_count_Solana2019-50-50_test.csv')