from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    # create pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('lda', LinearDiscriminantAnalysis()))
    model = Pipeline(estimators)
    # Prepare model

    # Evaluate pipeline
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


if __name__ == "__main__":
    main()