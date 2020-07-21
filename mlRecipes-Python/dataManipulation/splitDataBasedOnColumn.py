import pandas as pd
import os


def getClassData(file_path=None, new_file_path=None):
	"""
    Split file data based on class label and created a new file for each class

            Parameters:
                    :param file_path: a merged file, containing all classes
                    :param new_file_path: path to store new files

            Returns:
                    nothing
                    Saves new files at new_file_path

    """

	df = pd.read_csv(file_path)
	print(df.label.unique())
	label_list = df.label.unique()
	print(df.shape)
	for label in label_list:
		df1 = df[df.label == label]
		df1.drop(['label'], inplace=True, axis=1)
		file_name = label + '.csv'
		new_file_name = os.path.join(new_file_path, file_name)

		df1.to_csv(new_file_name, index=False)


if __name__ == '__main__':
	getClassData(
		file_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\merged_training_features_Solana2020a_70per_2019_70per.csv",
		new_file_path=r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-35i\similarity_score\features")