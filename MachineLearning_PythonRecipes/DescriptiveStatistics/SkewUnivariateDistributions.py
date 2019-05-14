from pandas import read_csv


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    skew= data.skew()
    print(skew)


if __name__ == "__main__":
    main()