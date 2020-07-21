import pandas as pd

def countClass(file_path=None):
	df = pd.read_csv(file_path)
	class_count = df.groupby(df.columns[-1]).size()
	print(class_count)


if __name__ == '__main__':
	countClass(
		file_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020\final_test_data\Solana2020-50-50\merged_labels_Solana2020a_50per.csv")