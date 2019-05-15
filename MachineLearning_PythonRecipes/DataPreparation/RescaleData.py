from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    array = data.values

    X = array[:, 0:8]
    Y = array[:, 8]

    scaler = MinMaxScaler(feature_range=(0,1))
    rescaledX = scaler.fit_transform(X)

    set_printoptions(precision=3)
    print(rescaledX[0:5, :])


if __name__ == "__main__":
    main()