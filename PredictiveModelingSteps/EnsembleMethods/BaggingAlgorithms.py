# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    kfold = KFold(n_splits=10, random_state=7)

    # Bagged Decision Tree
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
    result1= cross_val_score(model, X, Y, cv=kfold)
    print(result1.mean())

    # Random Forest
    num_trees = 100
    max_features = 3
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    result2 = cross_val_score(model, X, Y, cv=kfold)
    print(result2.mean())

    # Extra Tree
    num_trees = 100
    max_features = 7
    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
    result3 = cross_val_score(model, X, Y, cv=kfold)
    print(result3.mean())


if __name__ == "__main__":
    main()