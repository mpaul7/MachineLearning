from pandas import read_csv


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    # The read_csv() function returns a pandas DataFrame, which enables us to immediately start summarizing
    # and plotting data.
    data = read_csv(datafile, names=names)
    print(data.shape)
    print(data.head())

    url = 'https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv'
    data_url = read_csv(url, names=names)
    print(data_url.head())


if __name__ == "__main__":
    main()