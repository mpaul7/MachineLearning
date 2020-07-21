import os
import pandas as pd

def generateZScore(inputData=None):
	df = pd.read_csv(inputData)

	# ===== Labels ========
	class_path = r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\labels'
	zscore_path = r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\labels\zscore'

	# ===== features ========
	# class_path = r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features'
	# zscore_path = r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features\zscore'

	list_classes = df.actual_label.unique()
	for _class in list_classes:
		print(_class)
		file_name = _class + '.csv'
		df1 = df.loc[df['actual_label'] == _class]
		df1.drop(columns=['actual_label', 'pred_label'], inplace=True)
		file_path = os.path.join(class_path, file_name)
		df1.to_csv(file_path)
		cols = list(df1.columns)
		for col in cols:
			col_zscore = col + '_z'
			df1[col_zscore] = (df1[col] - df1[col].mean()) / df1[col].std(ddof=0)
		df1.drop(columns=cols, inplace=True)
		df1.drop(df1.columns[df1.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
		file_name = _class + '_zscore.csv'
		file_path = os.path.join(zscore_path, file_name)
		df1.to_csv(file_path)


if __name__ == '__main__':
	# ==== labels =========
	generateZScore(inputData=r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\merged-7class_test_feature-Solana2019a.csv")

	# ====== features ======
	# generateZScore(inputData=r"C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\merged_training_features_Solana2019a_70per_22-06-2020.csv")