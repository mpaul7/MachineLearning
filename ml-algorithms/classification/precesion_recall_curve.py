""":parameter
Link: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
Optimal Threshold for precision-Recall Curve
"""

# pr curve for logistic regression model
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot


def precisionRecallCurve():
	# generate dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
	# split into train/test sets
	trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
	# fit a model
	model = LogisticRegression(solver='lbfgs')
	model.fit(trainX, trainy)
	# predict probabilities
	yhat = model.predict_proba(testX)
	# keep probabilities for the positive outcome only
	yhat = yhat[:, 1]
	# calculate roc curves
	precision, recall, thresholds = precision_recall_curve(testy, yhat)
	# convert to f score
	fscore = (2 * precision * recall) / (precision + recall)
	# locate the index of the largest f score
	ix = argmax(fscore)
	print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
	# plot the roc curve for the model
	no_skill = len(testy[testy == 1]) / len(testy)
	pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	pyplot.plot(recall, precision, marker='.', label='Logistic')
	pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
	# axis labels
	pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
	pyplot.legend()
	# show the plot
	pyplot.show()


if __name__ == "__main__":
	precisionRecallCurve()
