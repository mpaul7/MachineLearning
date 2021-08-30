import dtale
import pandas as pd
import os

columns = ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', \
           'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb', \
           'min_itf', 'mean_itf', 'mx_itf', 'sdv_itf', \
           'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb', \
           'tot_dur', 'prot', 'num_pf', 'tot_bf', 'num_pb', 'tot_bb', \
           'tot_plf', 'tot_plb', \
           'fst_plsf', 'fst_plsb', 'num_plpf', 'num_plpb', \
           'ent_pb', 'ent_itb', 'actual_label']
def eda():
	df_b = pd.read_csv(r"C:\Users\Owner\projects3\Generalization\tasks\task-7\task-7-b\6Class_features_Solana2020a_80per.csv")
	df_b.columns = columns
	df_h = pd.read_csv(r"C:\Users\Owner\projects3\Generalization\tasks\task-7\task-7-h\6Class_features_Solana2020a_2019_80-5per.csv")
	df_h.columns = columns
	print(df_b)
	print(df_b.head())
	print(dtale.show(df_b))


if __name__ == "__main__":
	eda()