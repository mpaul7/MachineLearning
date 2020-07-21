from sklearn.feature_selection import SelectKBest, chi2
from pandas import read_csv
from numpy import set_printoptions


def main():
    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)

    set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    print(features[0:5, : ])


if __name__ == "__main__":
    main()