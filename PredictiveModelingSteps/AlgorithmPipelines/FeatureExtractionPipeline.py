from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def main():
    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    # create feature union
    features = []
    features.append(('pca', PCA(n_components=3)))
    features.append(('select_best', SelectKBest(k=6)))
    feature_union = FeatureUnion(features)

    # create pipeline
    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append(('logistic', LogisticRegression(solver='lbfgs')))
    model = Pipeline(estimators)
    # Prepare model

    # Evaluate pipeline
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


if __name__ == "__main__":
    main()