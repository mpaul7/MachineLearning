# Univariate Histograms

from matplotlib import pyplot as plt
from pandas import read_csv
import EasyTkinter as tk


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    data.plot(kind=' density ', subplots=True, layout=(3, 3), sharex=False)
  #  plt.hist(data)
   # data.hist()
    plt.show()


if __name__ == "__main__":
    main()