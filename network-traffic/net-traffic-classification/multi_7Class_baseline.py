# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:42:40 2019
@author: Yonglin Ren
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def read_train_test_from_files():
	# train_file_path = r"C:\Users\Owner\mltat\mltat_ext\resources\7Class-MN\merged-7class-trainingfile.csv"
	# test_file_path = r"C:\Users\Owner\mltat\mltat_ext\resources\7Class-MN\merged-7class-feats-labelsfile.csv"

	# # Task-16a
	# train_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-16a\final_test_data\merged_features_training_Solana2020c_70per.csv"
	# test_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-16a\final_test_data\merged_multi-4class-label_hdfs_Solana2020c-minus-missing-labels.csv"

	# Task-16b
	# train_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-16b\final_test_data\merged_features_4class_binary_training_Solana2020c_70per.csv"
	# test_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-16b\final_test_data\merged_label_hdfs_4Class_binary_v2.csv"

	# Task-17a
	# train_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17a\4Class-multi-features-training-Solana2020c.csv"
	# test_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17a\4Class-multi-feature-classifiation-result-file-hdfs_Solana2020c-minus-missing-labels-minus-lessthan-5packetcount.csv"

	# Task-17b
	# train_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17b\4Class-binary-feature-training-Solana2020c-70per.csv"
	# test_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17b\4Class-binary-feature-classification-result-hdfs-minus_missing-minus-lessthan-5packetcount.csv"

	# Task-17a-II
	# train_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17a-II\4Class-multi-features-training-Solana2020c.csv"
	# test_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17a-II\df_final.csv"

	# Task-17b-II
	train_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17b-II\4Class-binary-feature-training-Solana2020c-70per.csv"
	test_file_path = r"C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17b-II\df_final.csv"

	df_test = pd.read_csv(test_file_path)
	df_test_cols = df_test.columns.values.tolist()
	df_train = pd.read_csv(train_file_path, names=df_test_cols[:-1])
	return df_train, df_test


def cv_train(df_train, df_test):
	print("\nTraining starts: ")
	random_seed = 3572203868
	np.random.seed(random_seed)
	all_feats_names = ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', \
	                   'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb', \
	                   'min_itf', 'mean_itf', 'mx_itf', 'sdv_itf', \
	                   'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb', \
	                   'tot_dur', 'prot', 'num_pf', 'tot_bf', 'num_pb', 'tot_bb', \
	                   'tot_plf', 'tot_plb', \
	                   'fst_plsf', 'fst_plsb', 'num_plpf', 'num_plpb', \
	                   'ent_pb', 'ent_itb']
	target = 'actual_label'
	train_labels = df_train.iloc[:, -1]
	# print(train_labels)
	unique_classes = list(train_labels.unique())
	print(list(train_labels.unique()))
	train_data = df_train[all_feats_names]

	# split train and test for validation
	X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2)

	# use pipeline to do classification
	# c45 = DecisionTreeClassifier()
	# c45 = DecisionTreeClassifier(class_weight='balanced', random_state=random_seed)
	# c45 = DecisionTreeClassifier(random_state=random_seed, class_weight=class_weights)
	# c45 = RandomForestClassifier(n_estimators=100, random_state=random_seed)
	c45 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_seed)
	# c45 = AdaBoostClassifier(n_estimators=50, base_estimator=dtc, random_state=random_seed)

	pipe = Pipeline(memory=None, steps=[('scaler', StandardScaler()), ('clf', c45), ])
	parameters_grid = dict()
	gs = GridSearchCV(pipe, param_grid=parameters_grid, n_jobs=4)
	gs.fit(X_train, y_train)
	score = gs.score(X_test, y_test)
	df_to_test = df_test[all_feats_names].astype(np.float32)
	predicted_label = gs.predict(df_to_test)
	# df_prdict = pd.DataFrame(predicted_label, columns=['predicted_labels'])
	# df_compare = pd.concat([df_test, df_prdict], axis=1, sort=False)
	# df_compare.to_csv(r'C:\Users\Owner\mltat\Documents\mltat_ext\tasks\mltat_ext_7Class_predictions_baseline.csv')
	true_labels = df_test['actual_label'].values.tolist()

	# Evaluate
	conf_res = confusion_matrix(true_labels, predicted_label)
	print(conf_res)

	# ====================== 5 Class ==================================================================
	# columns = ['audio', 'file_transfer', 'network_service','social_media','video']
	# df_conf = pd.DataFrame(conf_res, columns=columns)
	# new_columns = ['audio', 'network_service', 'file_transfer', 'video', 'social_media']
	# df_conf_reorder = df_conf.reindex(columns=new_columns)
	# print(df_conf_reorder)
	# df_conf_reorder.to_csv(r'C:\Users\Owner\mltat\Documents\confusion_matrix\5conf1.csv', index=False)

	# ====================================== 7 Class ==================================================
	# columns = ['audio', 'audio_chat', 'network_service', 'p2p', 'social_media', 'video', 'video-chat']
	# df_conf = pd.DataFrame(conf_res, columns=columns)
	# new_columns = ['audio', 'audio_chat', 'network_service', 'video', 'video-chat', 'p2p', 'social_media']
	# df_conf_reorder = df_conf.reindex(columns=new_columns)
	# print(df_conf_reorder)
	# df_conf_reorder.to_csv(r'C:\Users\Owner\mltat\Documents\confusion_matrix\7conf1-task-58c-2.csv', index=False)

	# ====================================== 4 Class - Solana2020c ==================================================
	# columns = ['audio','file_transfer', 'network_service', 'p2p', 'video']
	# # columns = ['audio', 'file_transfer', 'p2p', 'video']
	# df_conf = pd.DataFrame(conf_res, columns=columns)
	# new_columns = ['audio', 'network_service', 'file_transfer', 'video',  'p2p']
	# # new_columns = ['audio', 'file_transfer', 'video', 'p2p']
	# df_conf_reorder = df_conf.reindex(columns=new_columns)
	# print(df_conf_reorder)
	# df_conf_reorder.to_csv(
	#     r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17a-II\task-17a_multi-conf_RF-100-balanced-baseline.csv',
	#     index=False)
	# ====================================== 4 Class binary - Solana2020c ==================================================
	columns = ['background', 'file_transfer', 'network_service']
	# columns = ['background', 'file_transfer']
	df_conf = pd.DataFrame(conf_res, columns=columns)
	new_columns = ['background', 'network_service', 'file_transfer']
	# new_columns = ['background', 'file_transfer']
	df_conf_reorder = df_conf.reindex(columns=new_columns)
	print(df_conf_reorder)
	df_conf_reorder.to_csv(
		r'C:\Users\Owner\mltat\data\mltat\TestFiles\train_test\Solana2020c\final_test_data\Solana2020c-70-30-v2\Task-17b-II\task-17b-II_binary-conf_RF-100-balanced-baseline.csv',
		index=False)


#
# multi_class_labels, multi_class_counts = np.unique(true_labels, return_counts=True)
# print(multi_class_labels)
# print(multi_class_counts)


def multi_class_c45_baseline_model_7class():
	df_train, df_test = read_train_test_from_files()
	cv_train(df_train, df_test)

# class_weights = {}
# for classes in unique_classes:
#     class_weights[classes] = df_train.shape[0] / (df_train.loc[df_train[target] == classes].shape[0]) * len(
#         unique_classes)
# class_weights = {'p2p': 1,
#                  'network_service': 1,
#                  'video': 1,
#                  'audio_chat': 1,
#                  'social_media': 1,
#                  'file_transfer': 10,
#                  'audio': 1}
# print(class_weights)