""":parameter
Resource: StatQuest
Link: https://www.youtube.com/watch?v=q90UDEgYqeI
predict: 'hd'
"""
import pandas as pd # to load and manipulate data and for one-hot encoding
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


def getData():
	"""	df columns
	['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'hd']
	"""
	df = pd.read_csv(r'C:\Users\Owner\DataScience\MachineLearning\resources\DataSets\processed.cleveland.csv')
	return df


def dt_clf():
	df = getData()
	print("df data types\n==================== \n {}".format(df.dtypes))

	""":parameter
	'thal' and 'ca' are object datatypes, and contain missing values. 
	"""
	# check for missing values
	print(df['ca'].unique())
	print(df['thal'].unique())
	""":parameter
	print the number of rows that contains missing values
	loc[], short for "location", let us specify which roes we want...
	and so we say we want any row with '?' in the column 'ca'
	OR
	any row with '?' in column 'thal'
	"""
	# Check how many rows contains missing values
	print("missing values \n===============\n {}".format(len(df.loc[(df['ca']=='?') | (df['thal']=='?')])))

	# remove missing values form the dataset
	df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]

	""":parameter
	Format Data Part 1: Split the data into Dependent and independent variables 
	"""
	X = df_no_missing.drop('hd', axis=1).copy()
	y = df_no_missing['hd'].copy()
	print(y)

	""":parameter
	We see that age, restbp, chol, and thalach are all flaot64, which is good, because we want them to be floationg point. 
	Sciokit-learn natively do not support categorical data., like cp contains 4 different categories. So we have use One-hot 
	encoding to convert a column of catagorical data into multiple columns of binary values. 
	"""
	X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])

	y_not_zero_index = y > 0
	y[y_not_zero_index] = 1

	""":parameter
	build A preliminary classification tree
	"""
	# split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

	# create a decision tree and fit it to the training data
	clf_dt = DecisionTreeClassifier(random_state=42)
	clf_dt = clf_dt.fit(X_train, y_train)

	#plot the tree
	plt.figure(figsize=(15, 7.5))
	plot_tree(clf_dt, filled=True, rounded=True, class_names=['No HD', 'Yes HD'], feature_names=X_encoded.columns)
	# plt.show()

	# plot confusion matrix
	# plot_confusion_matrix(clf_dt, X_test, y_test)

	"""":parameter
	Cost Complexity Pruning Part 1: Visualize alpha
	"""
	path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine the value for alpha
	ccp_alphas = path.ccp_alphas
	ccp_alphas = ccp_alphas[:-1]
	# print(ccp_alphas)

	clf_dts = []

	for ccs_alpha in ccp_alphas:
		clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccs_alpha)
		clf_dt.fit(X_train, y_train)
		clf_dts.append(clf_dt)

	train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
	test_Scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

	fig, ax = plt.subplots()
	ax.set_xlabel("alpha")
	ax.set_ylabel("accuracy")
	ax.set_title("Accuracy vs alpha for training and testing sets")
	ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
	ax.plot(ccp_alphas, test_Scores, marker='o', label="test", drawstyle="steps-post")
	ax.legend()
	plt.show()

	"""
	Cost Complexity pruning Part 2: Cross Validation for finding the Best Alpha
	"""
	clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016, )
	scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
	df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
	df.plot(x='tree', y='accuracy', marker='o', linestyle='--')
	# plt.show()

	alpha_loop_values = []
	for ccp_alpha in ccp_alphas:
		clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
		scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
		alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
	# print(alpha_loop_values)

	alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])

	alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
	# plt.show()

	ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014) & (alpha_results['alpha'] < 0.015)]['alpha']
	print(ideal_ccp_alpha)

	# convert ideal_alpha from a series to a float
	ideal_ccp_alpha = float(ideal_ccp_alpha)
	print(ideal_ccp_alpha)

	""":parameter
	Building, Evaluating, Drawing, and Interpreting the Final Classification Tree
	"""
	clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha= ideal_ccp_alpha)
	clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)
	disp = plot_confusion_matrix(clf_dt_pruned, X_test, y_test, display_labels=['Does not have HD', "Has HD"])
	disp.ax_.set_title("Pruned Confuaion matrix")
	print(disp.confusion_matrix)


	# Draw pruned tree
	plt.figure(figsize=(15, 7.5))
	plot_tree(clf_dt_pruned, filled=True, rounded=True, class_names=['No HD', 'Yes HD'], feature_names=X_encoded.columns)
	# plt.show()

if __name__ == "__main__":
	dt_clf()