from pandas import read_csv


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    # The read_csv() function returns a pandas DataFrame, which enables us to immediately start summarizing
    # and plotting data.
    data = read_csv(datafile, names=names)
    print(data.shape)
    print(data.head())


if __name__ == "__main__":
    main()