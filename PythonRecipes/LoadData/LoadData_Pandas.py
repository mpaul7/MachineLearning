import pandas as pd


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


def _load_all_data(file_patterns, columns=None):
    """Combine data from all files matching provided glob patterns into one pandas dataframe.
    Optionally filter on a subset of columns.

    Returns dataframe containing all data.
    """
    all_files = [f for pattern in file_patterns for f in glob.glob(pattern, recursive=True)]
    data = [pd.read_csv(d, usecols=columns, skipinitialspace=True) for d in all_files]
    df = pd.concat(data, ignore_index=True)
    return df

def test():
    _load_all_data(files)

if __name__ == "__main__":
    # main()
    test()