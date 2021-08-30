import numpy as np
import pandas as pd
import warnings
import os
import logging
import pickle
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def save_hgmm_models(models, filename="ext_trained_hgmm_model.sav"):
    with open(filename, 'wb') as f:
        logger.info("Saving model to %s" % filename)
        pickle.dump(models, f)


def load_hgmm_model(filename="ext_trained_hgmm_model.sav", packaged=True):
    if packaged:
        filename = find_hgmm_model_data(filename)

    with open(filename, 'rb') as f:
        logger.info("Loading model from %s" % filename)
        return pickle.load(f)


def find_hgmm_model_data(filename, packaged=True):

    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)


def getClassifier(clfname='gmmt_rf'):
    # dt = Pipeline([("data_imputation", DataImputation(categorical_features, feature_set = feature_names, subcategories = subcategories, normalize = normalize)),
    #                 ("encoder", DataEncoder(categorical_features)),
    #                 ('dt', DecisionTreeClassifier(class_weight = 'balanced', random_state = 10))])

    clfs = {}
    clfs['gmmt_rf'] = Pipeline([("gmmt", GMMTransformer(f_cpn=10, c_cpn=None, covar_type='tied',
                                                        class_gmm=False, fea_components=None, feature_gmm=True,
                                                        feature_by_class=True, recur=False, random_state=42,
                                                        fea_overall=None, std_scaler=True)),
                                ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))])

    clfs['gmmt_rft_rf'] = Pipeline([("gmmt", GMMTransformer(class_gmm=False, f_cpn=10, c_cpn=None,
                                                            covar_type='tied', fea_components=None, feature_gmm=True,
                                                            feature_by_class=True, recur=False, random_state=42,
                                                            fea_overall=None, std_scaler=True)),
                                    ("rft",
                                     RFTransformer(n_subfeatures=120, n_iters=100, binary_oa=False, max_output=False,
                                                   max_depth=None, resampling=False, random_state=42)),
                                    ('rf', RandomForestClassifier(class_weight=None, random_state=42))])

    clfs['st_gmmt_rft_rf'] = Pipeline([('st', StandardScaler()),
                                       ("gmmt", GMMTransformer(class_gmm=False, f_cpn=10, c_cpn=None,
                                                               covar_type='tied', fea_components=None, feature_gmm=True,
                                                               feature_by_class=True, recur=False, random_state=42,
                                                               fea_overall=None, std_scaler=True)),
                                       ("rft",
                                        RFTransformer(n_subfeatures=120, n_iters=1, binary_oa=False, max_output=False,
                                                      max_depth=None, resampling=False, random_state=42)),
                                       ('rf', RandomForestClassifier(class_weight=None, random_state=42))])

    return clfs[clfname]


class RFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_subfeatures=100, n_iters=1, binary_oa=True,
                 max_output=False, max_depth=None, resampling=False, random_state=42):
        self.n_subfeatures = n_subfeatures  # number of features for each sub model
        self.binary_oa = binary_oa  # build each sub model by binary oa
        self.resampling = resampling  # perfomr resampling for each sub model
        self.max_output = max_output  # max probability output
        self.n_iters = n_iters  # iteratively build sub models
        self.max_depth = max_depth  # max depth for each sub model
        self.random_state = random_state  # useful if resampling is False

        return

    def fit(self, X, y=None):

        self.classes_ = np.sort(np.unique(y))
        self.models_ = []
        self.subfeatures_ = []

        np.random.seed(self.random_state)

        if self.n_subfeatures is not None:
            self.n_subfeatures = min(self.n_subfeatures, X.shape[1])
            if self.resampling:
                np.random.seed(self.random_state)
            else:
                self.n_iters = int(X.shape[1] / self.n_subfeatures) + 1

        max_i = X.shape[1] - 1
        for k in range(self.n_iters):
            XX = X
            if self.n_subfeatures is not None:
                if self.resampling:
                    sampling_index = np.random.choice(X.shape[1], size=self.n_subfeatures, replace=False, p=None)
                else:
                    sampling_index = list(range(k * self.n_subfeatures, min((k + 1) * self.n_subfeatures, max_i)))

                if len(sampling_index) == 0:
                    continue

                XX = X[:, sampling_index]

            if self.binary_oa:
                for c in self.classes_:
                    yi = [0 if yi == c else 1 for yi in y]
                    clf = RandomForestClassifier(class_weight='balanced', max_depth=self.max_depth,
                                                 random_state=self.random_state)
                    clf.fit(XX, yi)

                    self.models_.append(clf)
                    if self.n_subfeatures is not None:
                        self.subfeatures_.append(sampling_index)
            else:
                clf = RandomForestClassifier(class_weight='balanced', max_depth=self.max_depth,
                                             random_state=self.random_state)
                clf.fit(XX, y)

                self.models_.append(clf)
                if self.n_subfeatures is not None:
                    self.subfeatures_.append(sampling_index)

            if self.n_subfeatures is None:
                break

        return self

    def transform(self, X):
        XX = None
        k = 0
        for clf in self.models_:
            if self.n_subfeatures is not None:
                subfeatures = self.subfeatures_[k]
                probs = clf.predict_proba(X[:, subfeatures])

            else:
                probs = clf.predict_proba(X)

            if self.max_output:
                probs = np.max(probs, axis=1).reshape(-1, 1)

            k += 1
            if XX is None:
                XX = probs
                continue

            XX = np.append(XX, probs, axis=1)
        # print('rft', XX.shape)
        return XX

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        params = {}
        params['n_iters'] = self.n_iters
        params['n_subfeatures'] = self.n_subfeatures
        params['binary_oa'] = self.binary_oa
        params['resampling'] = self.resampling
        params['max_output'] = self.max_output
        params['max_depth'] = self.max_depth
        params['random_state'] = self.random_state

        #        params['missing_value'] = self.missing_value
        return params

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)


class GMMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator='gmm', c_cpn=None, f_cpn=None, class_gmm=False, fea_components=None, feature_gmm=True,
                 covar_type='tied', feature_by_class=True, recur=False, reg_covar=1e-6, random_state=None,
                 fea_overall=None, std_scaler=None):
        self.estimator = estimator
        self.class_gmm = class_gmm
        self.feature_gmm = feature_gmm
        self.feature_by_class = feature_by_class
        self.fea_components = fea_components
        self.f_cpn = f_cpn
        self.c_cpn = c_cpn
        self.covar_type = covar_type
        self.recur = recur
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.fea_overall = fea_overall
        self.std_scaler = std_scaler  # if perform standard scaler by label before fit the model
        return

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            X = X.values

        self.model_type = GaussianMixture

        # classes_
        self.classes_ = None
        if y is not None:
            le = LabelEncoder().fit(y)
            self.classes_ = le.classes_  # np.sort(np.unique(y))

        self.fea_models_ = {}
        self.fea2_models_ = {}

        if self.classes_ is not None:
            if self.std_scaler:
                ss = StandardScaler()
                training_sets = [ss.fit_transform(X[y == yi]) for yi in self.classes_]  # standard scaler by label
            else:
                training_sets = [X[y == yi] for yi in self.classes_]  # split training data by label

            self.models_ = []
            if self.class_gmm:
                self.models_ = [
                    self.model_type(n_components=self.c_cpn, covariance_type=self.covar_type, reg_covar=self.reg_covar,
                                    random_state=self.random_state).fit(Xi)
                    for Xi in training_sets]  # train models per label

            if self.feature_gmm:
                fea_components = self.fea_components if self.fea_components is not None else [self.f_cpn] * X.shape[1]
                for yi in self.classes_:
                    if self.std_scaler:
                        Xi = ss.fit_transform(X[y == yi]) if self.feature_by_class else ss.fit_transform(X)  # split training data by label
                    else:
                        Xi = X[y == yi] if self.feature_by_class else X
                    self.fea_models_[yi] = []
                    for j in range(Xi.shape[1]):
                        Xij = Xi[:, j].reshape(-1, 1)
                        clf = self.model_type(n_components=fea_components[j], covariance_type=self.covar_type,
                                              reg_covar=self.reg_covar, random_state=self.random_state).fit(Xij)
                        self.fea_models_[yi].append(clf)

                    # self.fea2_models_[yi]=[]
                    # for i in range(Xi.shape[1]):
                    #     for j in range(i+1, Xi.shape[1]):
                    #         Xij=Xi[:, [i, j]] #.reshape(-1,1)
                    #         clf = self.model_type(n_components=fea_components[j], covariance_type=self.covar_type,reg_covar=self.reg_covar,random_state=self.random_state).fit(Xij)
                    #         self.fea2_models_[yi].append(clf)

                    if not self.feature_by_class:
                        break
        else:
            fea_components = self.fea_components if self.fea_components is not None else [self.f_cpn] * X.shape[1]
            self.fea_models_[0] = []
            for j in range(X.shape[1]):
                Xj = X[:, j].reshape(-1, 1)
                clf = self.model_type(n_components=fea_components[j], covariance_type=self.covar_type,
                                      reg_covar=self.reg_covar, random_state=self.random_state).fit(Xj)
                self.fea_models_[0].append(clf)

            # self.fea2_models_[0] = []
            # for i in range(X.shape[1]):
            #     for j in range(i + 1, X.shape[1]):
            #         Xj = X[:, [i, j]]  # .reshape(-1,1)
            #         clf = self.model_type(n_components=fea_components[j], covariance_type=self.covar_type,
            #                               reg_covar=self.reg_covar, random_state=self.random_state).fit(Xj)
            #         self.fea2_models_[0].append(clf)

        return self

    def transform(self, X, combine=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        XX = None
        if self.recur:
            XX = X

        if self.class_gmm:
            for model in self.models_:
                probs = model.predict_proba(X)
                if XX is None:
                    XX = probs
                    continue
                XX = np.append(XX, probs, axis=1)

        for yi in self.fea_models_:
            j = 0
            for clf in self.fea_models_[yi]:
                probs = clf.predict_proba(X[:, j].reshape(-1, 1))
                if XX is None:
                    XX = probs
                    continue

                XX = np.append(XX, probs, axis=1)
                j += 1

        for yi in self.fea2_models_:
            i, j = 0, 1
            rowlen = X.shape[1]
            for clf in self.fea2_models_[yi]:
                probs = clf.predict_proba(X[:, [i, j]])  # .reshape(-1,1))
                if XX is None:
                    XX = probs
                    continue

                XX = np.append(XX, probs, axis=1)
                j += 1
                if j >= rowlen:
                    i += 1
                    j = i + 1

        # print('gmmt', XX.shape)

        if self.fea_overall:
            final_clf = self.model_type(n_components=self.fea_overall, covariance_type=self.covar_type,
                                  reg_covar=self.reg_covar, random_state=self.random_state).fit(XX)
            overall_probs = final_clf.predict_proba(XX)

            return overall_probs

        else:
            return XX

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

    def get_params(self, deep=True):
        params = {}
        params['estimator'] = self.estimator
        params['class_gmm'] = self.class_gmm
        params['feature_gmm'] = self.feature_gmm
        params['feature_by_class'] = self.feature_by_class
        params['fea_components'] = self.fea_components
        params['c_cpn'] = self.c_cpn
        params['f_cpn'] = self.f_cpn
        params['covar_type'] = self.covar_type
        params['recur'] = self.recur
        params['reg_covar'] = self.reg_covar
        params['random_state'] = self.random_state

        return params


class collinear(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold):
        self.correlation_threshold = correlation_threshold
        return

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.corr_matrix = X.corr()
        # Extract the upper triangle of the correlation matrix
        upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > self.correlation_threshold)]
        self.features_to_drop = to_drop

        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Remove the features and return the data
        XX = X.drop(columns=self.features_to_drop)

        if XX.shape[1] == 0:
            return X

        # print(XX.shape)
        return XX

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        params = {}
        params['correlation_threshold'] = self.correlation_threshold
        #        params['unseen_missed'] = self.unseen_missed
        #        params['missing_value'] = self.missing_value
        return params

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)


class HGMMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, mt='gmm', covar_type='tied', n_cpn=20, c_cpn=20, f_cpn=10, n_pca=0,
                 reg_covar=1e-6, feature_by_class=True, fea_components=None, correlation_threshold=None,
                 n_subfeatures_for_rf=100, binary_ova_by_rf=False, feature_gmm=True, class_gmm=False,
                 recur=False, random_state=42):
        self.mt = mt
        self.model_type = GaussianMixture
        self.n_cpn = n_cpn  # not actually used
        self.c_cpn = c_cpn  # not actually used
        self.f_cpn = f_cpn  # number of components for each feature
        self.n_pca = n_pca  # not actually used
        self.feature_by_class = feature_by_class  # default is True; preferred
        self.fea_components = fea_components  # a list of component numbers for all features;
        # default is None, then f_cpn is given to all features;
        # prefer a pre-learned list of feature components
        self.covar_type = covar_type  # default is tied
        # self.n_features = n_features                #  not actually used
        self.feature_gmm = feature_gmm  # default is True
        self.class_gmm = class_gmm  # default is False; deprecated
        self.correlation_threshold = correlation_threshold  # default is None without collinear training; otherwise, fall in (0, 1)
        # only slight improvement for RF; usually for linear models
        self.n_subfeatures_for_rf = n_subfeatures_for_rf  # default is None for no use of rf ensemble; help build a powderful model;
        # for 100 rf, this value is proper between 100 and 150.
        self.binary_ova_by_rf = binary_ova_by_rf  # default is false;
        self.recur = recur  # not actually used
        self.reg_covar = reg_covar  # default
        self.random_state = random_state  # default

    def fit(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.values

        # classes_
        le = LabelEncoder().fit(y)
        self.classes_ = le.classes_  # np.sort(np.unique(y))

        # if self.n_subfeatures_for_rf is not None:
        #     self.rf_transformer = RFTransformer(n_subfeatures = self.n_subfeatures_for_rf,
        #                                         random_state=self.random_state)
        #     self.rf_transformer.fit(X, y)
        #     X = self.rf_transformer.transform(X)

        training_sets = [X[y == yi] for yi in self.classes_]  # split training data by label

        self.models_ = []
        if self.class_gmm:
            self.models_ = [
                self.model_type(n_components=self.c_cpn, covariance_type=self.covar_type, reg_covar=self.reg_covar,
                                random_state=self.random_state).fit(Xi)
                for Xi in training_sets]  # train models per label

        self.fea_models_ = {}
        self.fea2_models_ = {}
        if self.feature_gmm:
            fea_components = self.fea_components if self.fea_components is not None else [self.f_cpn] * X.shape[1]
            for yi in self.classes_:
                Xi = X[y == yi] if self.feature_by_class else X
                self.fea_models_[yi] = []
                for j in range(Xi.shape[1]):
                    Xij = Xi[:, j].reshape(-1, 1)
                    clf = self.model_type(n_components=fea_components[j], covariance_type=self.covar_type,
                                          reg_covar=self.reg_covar, random_state=self.random_state).fit(Xij)
                    self.fea_models_[yi].append(clf)

                # self.fea2_models_[yi]=[]
                # for i in range(Xi.shape[1]):
                #     for j in range(i+1, Xi.shape[1]):
                #         Xij=Xi[:, [i, j]] #.reshape(-1,1)
                #         clf = self.model_type(n_components=fea_components[j], covariance_type=self.covar_type,reg_covar=self.reg_covar,random_state=self.random_state).fit(Xij)
                #         self.fea2_models_[yi].append(clf)

                if not self.feature_by_class:
                    break

        # self.X_final_train = self.predict_proba(X)  # get score_sample from each model as the final model training data
        # self.f_model = self.model_type()
        # params = {'n_components': len(np.unique(y)),
        #           'covariance_type': self.covar_type}
        # self.f_model.set_params(**params)

        # self.f_model.means_init = np.array([self.X_final_train[y == yi].mean(axis=0)
        #                                     for yi in self.classes_])  # get mean by column per class as final model parameter
        # # print(self.f_model.means_init)

        # self.f_model.fit(self.X_final_train)  # Not sure if we need to fit with y, supervised GMM example doesn't fit with y

        # print(X.shape)
        X_final_train = self.proba_outputs(X)
        # print(X_final_train.shape)

        if self.correlation_threshold is not None:
            self.col_remover = collinear(self.correlation_threshold).fit(X_final_train)
            X_final_train = self.col_remover.transform(X_final_train)
            # print(X_final_train.shape)

        if self.n_subfeatures_for_rf is not None or self.binary_ova_by_rf:
            self.rf_transformer = RFTransformer(n_subfeatures=self.n_subfeatures_for_rf,
                                                binary_oa=self.binary_ova_by_rf,
                                                n_iters=10,
                                                random_state=self.random_state)
            self.rf_transformer.fit(X_final_train, y)
            X_final_train = self.rf_transformer.transform(X_final_train)
            # print(X_final_train.shape)

        # n_features = min(self.n_features, X_final_train.shape[1])
        # self.f_model = RFE(estimator=DecisionTreeClassifier(random_state=self.random_state), n_features_to_select=n_features, step=1)
        print('training rf...')
        self.f_model = RandomForestClassifier(random_state=self.random_state)
        self.f_model.fit(X_final_train, y)
        return self

    # def predict_proba(self, X):
    #     logprobs = np.vstack([model.score_samples(X)
    #                           for model in self.models_]).T

    def proba_outputs(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        XX = None
        if self.recur:
            XX = X

        if self.class_gmm:
            for model in self.models_:
                probs = model.predict_proba(X)
                if XX is None:
                    XX = probs
                    continue
                XX = np.append(XX, probs, axis=1)

        for yi in self.fea_models_:
            j = 0
            for clf in self.fea_models_[yi]:
                probs = clf.predict_proba(X[:, j].reshape(-1, 1))
                if XX is None:
                    XX = probs
                    continue

                XX = np.append(XX, probs, axis=1)
                j += 1

        for yi in self.fea2_models_:
            i, j = 0, 1
            rowlen = X.shape[1]
            for clf in self.fea2_models_[yi]:
                probs = clf.predict_proba(X[:, [i, j]])  # .reshape(-1,1))
                if XX is None:
                    XX = probs
                    continue

                XX = np.append(XX, probs, axis=1)
                j += 1
                if j >= rowlen:
                    i += 1
                    j = i + 1

        if self.n_pca > 0:
            XX = XX if XX is not None else X
            ncomp = min(self.n_pca, XX.shape[1])
            self.pca = PCA(n_components=self.n_pca).fit(XX)
            XX = self.pca.transform(XX)

        return XX if XX is not None else X

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            # if self.n_subfeatures_for_rf is not None:
        #     X = self.rf_transformer.transform(X)

        X_final_test = self.proba_outputs(X)  # get score_sample from each model as the final model testing data
        if self.self.correlation_threshold is not None:
            X_final_test = self.col_remover.transform(X_final_test)
        if self.n_subfeatures_for_rf is not None or self.binary_ova_by_rf:
            X_final_test = self.rf_transformer.transform(X_final_test)
        return self.f_model.predict_proba(X_final_test)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            # if self.n_subfeatures_for_rf is not None:
        #     X = self.rf_transformer.transform(X)
        X_final_test = self.proba_outputs(X)  # get score_sample from each model as the final model testing data
        if self.correlation_threshold is not None:
            X_final_test = self.col_remover.transform(X_final_test)
        if self.n_subfeatures_for_rf is not None or self.binary_ova_by_rf:
            X_final_test = self.rf_transformer.transform(X_final_test)
        return self.f_model.predict(X_final_test)

    # must implementation

    def score(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            # if self.n_subfeatures_for_rf is not None:
        #     X = self.rf_transformer.transform(X)
        X_final_test = self.proba_outputs(X)
        if self.self.correlation_threshold is not None:
            X_final_test = self.col_remover.transform(X_final_test)
        if self.n_subfeatures_for_rf is not None or self.binary_ova_by_rf:
            X_final_test = self.rf_transformer.transform(X_final_test)
        return self.f_model.score(X)

    def score_samples(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            # if self.n_subfeatures_for_rf is not None:
        #     X = self.rf_transformer.transform(X)
        X_final_test = self.proba_outputs(X)
        if self.self.correlation_threshold is not None:
            X_final_test = self.col_remover.transform(X_final_test)
        if self.n_subfeatures_for_rf is not None or self.binary_ova_by_rf:
            X_final_test = self.rf_transformer.transform(X_final_test)
        return self.f_model.score_samples(X_final_test)

    def get_params(self, deep=True):
        params = {}
        params['mt'] = self.mt
        params['model_type'] = self.model_type
        params['n_cpn'] = self.n_cpn
        params['c_cpn'] = self.c_cpn
        params['f_cpn'] = self.f_cpn
        params['n_pca'] = self.n_pca
        params['feature_gmm'] = self.feature_gmm
        params['feature_by_class'] = self.feature_by_class
        params['fea_components'] = self.fea_components
        # params['n_features'] = self.n_features
        params['n_subfeatures_for_rf'] = self.n_subfeatures_for_rf
        params['binary_ova_by_rf'] = self.binary_ova_by_rf
        params['covar_type'] = self.covar_type
        params['reg_covar'] = self.reg_covar
        params['collinear_removing'] = self.collinear_removing
        params['recur'] = self.recur
        params['random_state'] = self.random_state
        # params['verbose']  = self.verbose
        return params

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self