
import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment', None)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# from labeler.ui.filter_widget import MessageBox

# Log transform these
LOG_FEATURES = ['min_itf', 'mean_itf', 'mx_itf', 'sdv_itf',
           'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb',
           'tot_dur',
           'num_pf', 'tot_bf', 'num_pb', 'tot_bb', 'tot_plf', 'tot_plb','num_plpb'] # forward is relatively small


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_feature):
        self.log_feature = log_feature

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Lognorm
        XX = X.copy()
        for lf in self.log_feature:
            XX[lf] = np.log(X[lf] + 1e-9)
        return XX

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class PreprocessorPipelineFactory():

    def get_pipeline(self, procname=None):
        preproc = {}

        # preproc['st_gmmt'] = Pipeline([('st', StandardScaler()),
        #                                ("gmmt", GMMTransformer(class_gmm=False, f_cpn=5, c_cpn=None, covar_type='tied',
        #                                         fea_components=[1,4,5,4,5], feature_gmm=True, feature_by_class=True,
        #                                         recur=False, random_state=42, fea_overall=None, std_scaler=None)),
        #                                ])

        preproc['log_minmax'] = Pipeline([
                                         ('log', LogTransformer(log_feature=LOG_FEATURES)),
                                         ('minmax', MinMaxScaler()),
                                        ])
        preproc['log_stdscaler'] = Pipeline([
                                            ('log', LogTransformer(log_feature=LOG_FEATURES)),
                                            ('stdScaler', StandardScaler()),
                                            ])
        preproc['stdscaler'] = Pipeline(
                                        [('stdScaler', StandardScaler())
                                         ])
        preproc['minmax'] = Pipeline(
                                    [('minmax', MinMaxScaler())
                                     ])

        preproc['log'] = Pipeline([
            ('log', LogTransformer(log_feature=LOG_FEATURES))
            ])

        # preproc['log_st_gmmt'] = Pipeline([
        #                                  ('log', LogTransformer(log_feature=LOG_FEATURES)),
        #                                  ('st', StandardScaler()),
        #                                  ('gmmt', GMMTransformer(class_gmm=False, f_cpn=11, c_cpn=None, covar_type='tied',
        #                                           fea_components=[4,5,2,3,2,3,1,4,5,4,5], feature_gmm=True, feature_by_class=True,
        #                                           recur=False, random_state=42, fea_overall=None, std_scaler=None)),
        #                                  ])

        # preproc['gmmt'] = Pipeline([
        #     ('gmmt', GMMTransformer(class_gmm=False, f_cpn=11, c_cpn=None, covar_type='tied',
        #                             fea_components=[4, 5, 2, 3, 2, 3, 1, 4, 5, 4, 5], feature_gmm=True,
        #                             feature_by_class=True,
        #                             recur=False, random_state=42, fea_overall=None, std_scaler=None)),
        # ])

        return preproc[procname]
