from sklearn.preprocessing import Normalizer
from pandas import read_csv
from numpy import set_printoptions

def main():
    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    scaler = Normalizer().fit(X)
    NormalizedX  = scaler.transform(X)

    set_printoptions(precision=3)
    print(NormalizedX[0:5, :])


if __name__ == "__main__":
    main()