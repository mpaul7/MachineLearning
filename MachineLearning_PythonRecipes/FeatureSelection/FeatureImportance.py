from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from pandas import read_csv


def main():
    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    model = ExtraTreesClassifier(n_estimators=10) # to supress Future Warnings
    model.fit(X, Y)
    print(model.feature_importances_)


if __name__ == "__main__":
    main()