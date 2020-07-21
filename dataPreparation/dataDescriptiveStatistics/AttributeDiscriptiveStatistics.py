from pandas import read_csv
from pandas import set_option


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    set_option('display.width', 100)
    set_option('precision', 3)
    description = data.describe()
    print(description)


if __name__ == "__main__":
    main()