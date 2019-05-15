from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from pandas import read_csv


def main():
    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    model = LogisticRegression(solver='lbfgs') # to supress the Future Warning
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print(fit.n_features_)
    print(fit.support_)
    print(fit.ranking_)
    #print("Num Features: %d") % fit.n_features_
    #print("Selected Features: %s") % fit.support_
    #print("Feature Ranking: %s") % fit.ranking_


if __name__ == "__main__":
    main()