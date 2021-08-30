import pandas as pd
import numpy as np


def prepare_conf_matrix():
	print("hello")
	df = pd.read_csv(r"C:\Users\Owner\projects3\twc\cpkt\cpkt\twc\twc\results\task4a-3.csv", header=None)
	print(df)
	# print(df.iloc[:,2].sum())
	# print(df.iloc[2,2])
	precision=[]
	recall = []
	total_flows = []
	weighted_presision = []
	tpv=[]
	col_index = [i for i in range(0, df.shape[1])]
	print(col_index)
	for i in col_index:
		# print(i)
		# precision.append(df.iloc[i, i]/df.iloc[:, i].sum())
		recall.append(df.iloc[i, i]/df.iloc[i, :].sum())
		print(df.iloc[i, :].sum())
		print(df.iloc[i, i]/(df.iloc[i, col_index[i]])*(df.iloc[i, :].sum()/df.iloc[i, :].sum())+
		                    df.iloc[i, i+1])
		tpv.append(df.iloc[i, i])

	total_rate = sum(tpv)/df.values.sum()

	# print(tpv)
	# print(sum(tpv))
	# print(df.values.sum())
	print(total_rate)


if __name__ == '__main__':
	prepare_conf_matrix()