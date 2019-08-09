from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV


def main():

    datafile = '~/Dropbox/Workspaces/MachineLearning/Resources/DataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    alphas = numpy.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    param_grid = dict(alpha=alphas)
    model = Ridge()
    research = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7 )
    research.fit(X, Y)
    print(research.best_score_)
    print(research.best_estimator_.alpha)


if __name__ == "__main__":
    main()