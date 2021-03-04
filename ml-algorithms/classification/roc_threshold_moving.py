""":parameter
A Gentle Introduction to Threshold-Moving for Imbalanced Classification
Link: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
"""

# roc curve for logistic regression model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax


""" # generate dataset
to create a synthetic binary classification problem with 10,000 examples (rows), 99 percent of which belong to the 
majority class and 1 percent belong to the minority class.
"""
def roc_LogR():
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	                           n_clusters_per_class=1, weights=[0.99], flip_y=0,
	                           random_state=4)
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
	fpr, tpr, thresholds = roc_curve(testy, yhat)
	# calculate the g-mean for each threshold
	gmeans = sqrt(tpr * (1 - fpr))
	# locate the index of the largest g-mean
	ix = argmax(gmeans)
	print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
	# plot the roc curve for the model
	pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
	pyplot.plot(fpr, tpr, marker='.', label='Logistic')
	pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
	# axis labels
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	pyplot.legend()
	# show the plot
	pyplot.show()


if __name__ == "__main__":
	roc_LogR()