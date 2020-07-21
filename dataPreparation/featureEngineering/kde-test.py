import os
from math import log2
import numpy as np
from statistics import mean, stdev
import pandas as pd
from matplotlib import pyplot
import seaborn as sns
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott, select_bandwidth
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import datetime
sns.set()


def listFiles(paths=None):
    files = []
    list_paths = []
    # r=root, d=directories, f = files
    # print(paths)
    for path in paths:
        for p, d, f in os.walk(path):
            list_paths.append(p)
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(p, file))
    # files.append(file)
    return files


def outlierDetectionRemoval(data=None):
    data_mean, data_std = mean(data), stdev(data)
    # identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # identify outliers
    outliers = [x for x in data if x < lower or x > upper]
    # print(outliers)
    # print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    # print(type(outliers_removed))
    data_df = pd.Series(outliers_removed)
    # print('Non-outlier observations: %d' % len(outliers_removed))
    return data_df

def kde_test():
    start = datetime.datetime.now()
    print(start)
    files_features = listFiles(paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\features'])
    files_labels = listFiles(paths=[r'C:\Users\Owner\mltat\data\mltat\TestFiles\feature_engineering\similarity_score\labels'])
    dfX = pd.read_csv(files_features[0])
    dfY = pd.read_csv(files_labels[0])
    _featureX = dfX[dfX.columns[2]]
    _featureY = dfY[dfY.columns[2]]
    data = outlierDetectionRemoval(_featureX).to_numpy()
    # featureY = outlierDetectionRemoval(_featureY).to_numpy()


    silverman_bandwidth = bw_silverman(data)

    print(f"Silverman bandwidth = {silverman_bandwidth}")
    print('Total time {}'.format((datetime.datetime.now()) - start))
    # select bandwidth allows to set a different kernel
    silverman_bandwidth_gauss = select_bandwidth(data, bw='silverman', kernel='gauss')
    scott_bandwidth = bw_scott(data)
    print(f"Scott bandwidth = {scott_bandwidth}")
    print('Total time {}'.format((datetime.datetime.now()) - start))
    model = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 15, 50)}, cv=5)
    model.fit(data[:, None])
    # model.best_params_['bandwidth']

    cv_bandwidth = model.best_params_
    print(f"Silverman bandwidth = {silverman_bandwidth}")
    print(f"Scott bandwidth = {scott_bandwidth}")
    print(f"CV bandwidth = {cv_bandwidth}")
    end = datetime.datetime.now()

    print('Total time {}'.format(end-start))


def test():
    max = 100
    max2 = min([max, 100000])
    print(max2)


if __name__ == '__main__':
    # kde_test()
    test()