#The most common method for calculating correlation is Pearson’s Correlation Coefficient
# It assumes a normal distribution of the attributes involved.
# value = -1 > negative  correlation
# value =  1 > positive  correlation
# value = -1 > no  correlation
# Some ML algorithms like linear and logistic regression can suffer poor performance if there are highly correlated attributes in your dataset.
# So, it is a good idea to review all of the pairwise correlations of the attributes in your dataset.

from pandas import read_csv
from pandas import set_option


def main():

    datafile = '/home/mpaul/Dropbox/Professional/dataScientist/dataSets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres ', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(datafile, names=names)
    set_option('display.width', 100)
    set_option('precision', 3)
    correlations = data.corr('pearson')
    print(correlations)


if __name__ == "__main__":
    main()