
import os

def splitTrainTest(dataDetails=None,
                    _class=None,
                    feat_train_path=None,
                    feat_test_path=None,
                    percentage=None):
	max_flow_count = dataDetails['cumm_flow_count'].iloc[-1]

	# ==== Features =================
	df_train = dataDetails.loc[dataDetails['cumm_flow_count'] <= max_flow_count * (percentage / 100)]
	train_FeatureFile = _class + '-' + str(percentage) + '.csv'
	train_FeatureFile_path = os.path.join(feat_train_path, train_FeatureFile)
	df_train.to_csv(train_FeatureFile_path, index=False)

	df_test = dataDetails.loc[dataDetails['cumm_flow_count'] >= max_flow_count * (percentage / 100)]
	df_test['cumm_flow_count'] = df_test['birDir_flow_count'].cumsum()
	df_test['cumm_pcap_size_mb'] = df_test['pcap_size_mb'].cumsum()
	test_FeatureFile = _class + '-' + str(100 - percentage) + '.csv'
	test_FeatureFile_path = os.path.join(feat_test_path, test_FeatureFile)
	df_test.to_csv(test_FeatureFile_path, index=False)