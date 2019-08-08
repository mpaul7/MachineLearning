# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]
    kfold = KFold(n_splits=10, random_state=7)

    # create the sub models
    estimators = []
    model1 = LogisticRegression(solver='lbfgs')
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC(gamma='scale')
    estimators.append(('svm', model3))

    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = cross_val_score(ensemble, X, Y, cv=kfold)
    print(results.mean())


if __name__ == "__main__":
    main()