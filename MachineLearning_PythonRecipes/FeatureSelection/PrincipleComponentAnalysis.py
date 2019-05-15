from sklearn.decomposition import PCA
from pandas import read_csv


def main():
    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    pca = PCA(n_components=3)
    fit = pca.fit(X)
    print(fit.explained_variance_ratio_)
    print(fit.components_)


if __name__ == "__main__":
    main()