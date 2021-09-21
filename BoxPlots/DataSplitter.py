import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from datetime import datetime
import numpy as np
import os

# For stratified Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold

# For imputing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

# For Encoding and Preprocessing
from sklearn import preprocessing

# For feature selection
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.feature_selection import f_classif

# For Outlier Detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

date = datetime.today().strftime('%m%d%y')  # For labelling purposes


class dataSplitter():

    def __init__(self, PARAMS, path, name=None):
        self.PARAMS = PARAMS
        self.data = pd.read_csv(path)
        self.name = name

    def prepData(self):

        self.split1 = 5
        self.split2 = 107

        X = self.data.drop(columns='Preeclampsia/Eclampsia')
        Y = self.data['Preeclampsia/Eclampsia']

        return X, Y

    def setData(self, X_train, X_test, Y_train, Y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def scaleData(self):

        if self.PARAMS['Normalize'] == 'MinMax':
            scaler = MinMaxScaler()
        elif self.PARAMS['Normalize'] == 'StandardScale':
            scaler = StandardScaler()
        elif self.PARAMS['Normalize'] == 'Robust':
            scaler = RobustScaler()

        # Fit and transform training data, then transform val and test using info gained from fitting
        scaleColumns = ['MotherAge', 'WeightAtAdmission',
                        'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions', 'WeightAtAdmission',
                        'PNV_GestAge', 'PNV_Weight_Oz', 'MAP', 'Prev_highBP']

        self.X_train[scaleColumns] = scaler.fit_transform(self.X_train[scaleColumns])
        self.X_val[scaleColumns] = scaler.transform(self.X_val[scaleColumns])
        self.X_test[scaleColumns] = scaler.transform(self.X_test[scaleColumns])

    def normalizeData(self):

        if self.PARAMS['Normalize'] == 'MinMax':
            scaler = MinMaxScaler()
        elif self.PARAMS['Normalize'] == 'StandardScale':
            scaler = StandardScaler()

        # Fit and transform training data, then transform val and test using info gained from fitting
        scaleColumns = ['MotherAge', 'WeightAtAdmission',
                        'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions', 'WeightAtAdmission',
                        'PNV_GestAge', 'PNV_Weight_Oz', 'MAP', 'Prev_highBP']

        self.X_train[scaleColumns] = scaler.fit_transform(self.X_train[scaleColumns])
        self.X_test[scaleColumns] = scaler.transform(self.X_test[scaleColumns])

    def encodeData(self):

        encodeCols = ['Insurance', 'MaternalNeuromuscularDisease', 'MCollagenVascularDisease',
                      'MStructuralHeartDiseas', 'MPostPartumComplications', 'DiabetesMellitus', 'ThyroidDisease',
                      'MLiverGallPanc', 'KidneyDisease', 'MAnemiaWOHemoglobinopathy', 'MHemoglobinopathy',
                      'Thrombocytopenia', 'ViralOrProtoInf', 'OtherSubstanceAbuse', 'InfSex', 'CNSAbnormality',
                      'RaceCollapsed']

        selectCat = [c for c in self.X_train.columns if (c in encodeCols)]

        ohe = OneHotEncoder(handle_unknown='ignore')

        # Train on the categorical variables
        ohe.fit(self.data[selectCat])

        X_trainCodes = ohe.transform(self.X_train[selectCat]).toarray()
        X_testCodes = ohe.transform(self.X_test[selectCat]).toarray()
        feature_names = ohe.get_feature_names(selectCat)

        self.X_train = pd.concat([self.X_train.drop(columns=selectCat).reset_index(drop=True),
                                  pd.DataFrame(X_trainCodes, columns=feature_names).astype(int).reset_index(drop=True)],
                                 axis=1)


        self.X_test = pd.concat([self.X_test.drop(columns=selectCat).reset_index(drop=True),
                                 pd.DataFrame(X_testCodes, columns=feature_names).astype(int).reset_index(drop=True)],
                                axis=1)

        # OHE adds unnecessary nan column, which needs to be dropped
        self.X_train = self.X_train.loc[:, ~self.X_train.columns.str.endswith('_nan')]
        self.X_test = self.X_test.loc[:, ~self.X_test.columns.str.endswith('_nan')]

    def imputeData(self, data1=None, data2=None):
        # Scikitlearn's Iterative imputer
        # Default imputing method is Bayesian Ridge Regression

        if self.PARAMS['estimator'] == "BayesianRidge":
            estimator = BayesianRidge()
        elif self.PARAMS['estimator'] == "DecisionTree":
            estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0)
        elif self.PARAMS['estimator'] == "ExtraTrees":
            estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
        elif self.PARAMS['estimator'] == "KNN":
            estimator = KNeighborsRegressor(n_neighbors=15)

        MI_Imp = IterativeImputer(random_state=0, estimator=estimator)

        if (self.name == 'MOMI'):
            if self.data.isnull().values.any():
                self.X_train_imputed = pd.DataFrame(MI_Imp.fit_transform(self.X_train), columns=self.X_train.columns)
                self.X_test_imputed = pd.DataFrame(MI_Imp.transform(self.X_test), columns=self.X_test.columns)

                # Rounding only the categorical variables that were imputed
                self.X_train = self.X_train_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
                                                           'DeliveriesPriorAdmission': 0, 'TotalAbortions': 0,
                                                           'Primagrivada': 0, 'MaternalNeuromuscularDisease': 0,
                                                           'KidneyDisease': 0, 'Thrombocytopenia': 0, 'InfSex': 0,
                                                           'CNSAbnormality': 0, 'CongenitalSyphilis': 0, 'UTI': 0,
                                                           'RaceCollapsed': 0, 'Systolic': 0})

                self.X_test = self.X_test_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
                                                         'DeliveriesPriorAdmission': 0, 'TotalAbortions': 0,
                                                         'Primagrivada': 0, 'MaternalNeuromuscularDisease': 0,
                                                         'KidneyDisease': 0, 'Thrombocytopenia': 0, 'InfSex': 0,
                                                         'CNSAbnormality': 0, 'CongenitalSyphilis': 0, 'UTI': 0,
                                                         'RaceCollapsed': 0, 'Systolic': 0})

            # Fix incorrectly imputed value
            self.X_train['RaceCollapsed'] = np.where(((self.X_train['RaceCollapsed'] > 4)), 4,
                                                     self.X_train['RaceCollapsed'])

            self.X_test['RaceCollapsed'] = np.where(((self.X_test['RaceCollapsed'] > 4)), 4,
                                                    self.X_test['RaceCollapsed'])

            self.X_train[self.X_train < 0] = 0
            self.X_test[self.X_test < 0] = 0

        else:
            if (self.data.isnull().values.any() == True):
                self.X_train = pd.DataFrame(MI_Imp.fit_transform(self.X_train), columns=self.X_train.columns)
                self.X_test = pd.DataFrame(MI_Imp.transform(self.X_test), columns=self.X_test.columns)

                self.X_train = self.X_train.round()
                self.X_test = self.X_test.round()

    def detectOutliers(self, con='auto'):

        print(self.X_train.shape, self.Y_train.shape)

        if self.PARAMS['OutlierRemove'] == 'iso':
            out = IsolationForest(contamination=con)
        elif self.PARAMS['OutlierRemove'] == 'lof':
            out = LocalOutlierFactor(contamination=con)
        elif self.PARAMS['OutlierRemove'] == 'ocsvm':
            out = OneClassSVM(nu=0.01)
        elif self.PARAMS['OutlierRemove'] == 'ee':
            out = EllipticEnvelope()

        yhat = out.fit_predict(self.X_train)

        # select all rows that are not outliers
        mask = (yhat != -1)

        self.X_train = self.X_train.loc[mask]
        self.Y_train = self.Y_train.loc[mask]

        print(self.X_train.shape, self.Y_train.shape)

    def featureSelection(self):

        # If there are less features than the number selected
        numFeatures = min(self.PARAMS['Feature_Num'], (self.X_train.shape[1]))

        if self.PARAMS['Feature_Selection'] == "XGBoost":
            model = XGBClassifier()
            model.fit(self.X_train, self.Y_train)

            # Save graph
            ax = plot_importance(model, max_num_features=numFeatures)
            fig1 = pyplot.gcf()
            # pyplot.show()

            fig1.savefig('XGBoostTopFeatures.png', bbox_inches='tight')

            # Get and save best features
            feature_important = model.get_booster().get_fscore()
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
            topFeatures = list(data.index[0:numFeatures])

        if self.PARAMS['Feature_Selection'] == "Chi2":
            # instantiate SelectKBest to determine 20 best features
            fs = SelectKBest(score_func=chi2, k=numFeatures)
            fs.fit(self.X_train, self.Y_train)
            df_scores = pd.DataFrame({"Scores": fs.scores_, "P-values": fs.pvalues_})
            df_columns = pd.DataFrame(self.X_train.columns)
            # concatenate dataframes
            feature_scores = pd.concat([df_columns, df_scores], axis=1)
            feature_scores.columns = ['Feature_Name', 'Chi2 Score', 'P-value']  # name output columns
            feature_scores.sort_values(by=['Chi2 Score'], ascending=False, inplace=True)
            features = feature_scores.iloc[0:numFeatures]
            chi2Features = features['Feature_Name']
            topFeatures = list(chi2Features)
            self.Chi2Features = features

        if self.PARAMS['Feature_Selection'] == "MI":
            # Mutual Information features
            fs = SelectKBest(score_func=mutual_info_classif, k=numFeatures)
            fs.fit(self.X_train, self.Y_train)
            df_scores = pd.DataFrame(fs.scores_)
            df_columns = pd.DataFrame(self.X_train.columns)
            # concatenate dataframes
            feature_scores = pd.concat([df_columns, df_scores], axis=1)
            feature_scores.columns = ['Feature_Name', 'MI Score']  # name output columns
            feature_scores.sort_values(by=['MI Score'], ascending=False, inplace=True)
            features = feature_scores.iloc[0:numFeatures]
            mutualInfoFeatures = features['Feature_Name']
            topFeatures = list(mutualInfoFeatures)
            self.MIFeatures = features

        if self.PARAMS['Feature_Selection'] == None:
            topFeatures = self.X_train.columns

        self.X_train = self.X_train[topFeatures]
        self.X_test = self.X_test[topFeatures]

    def saveData(self, counter):

        self.X_train.to_csv('Data/' + self.PARAMS['dataset'] + '/' + date + '_X_Train_' + str(counter) + '.csv', index=False)
        self.Y_train.to_csv('Data/' + self.PARAMS['dataset'] + '/' + date + '_Y_Train_' + str(counter) + '.csv', index=False)
        self.X_test.to_csv('Data/' + self.PARAMS['dataset'] + '/' + date + '_X_Test_' + str(counter) + '.csv', index=False)
        self.Y_test.to_csv('Data/' + self.PARAMS['dataset'] + '/' + date + '_Y_Test_' + str(counter) + '.csv', index=False)


if __name__ == "__main__":

    PARAMS = {'estimator': "BayesianRidge",
              'Normalize': 'MinMax',
              'OutlierRemove': 'lof',
              'Feature_Selection': 'Chi2',
              'Feature_Num': 20,
              'TestSplit': 0.10,
              'ValSplit': 0.10,
              'dataset': 'Texas'}

    # Get path to cleaned data
    parent = os.path.dirname(os.getcwd())
    path = os.path.join(parent, 'Data/Processed/Texas/Full/Outliers/Complete/Chi2_Categorical_041521.csv')


    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)
    splitter = dataSplitter(PARAMS, path, name='Texas')
    X, y = splitter.prepData()

    xYDict = {}
    counter = 1

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        splitter.setData(X_train, X_test, y_train, y_test)
        splitter.imputeData()
        #splitter.detectOutliers()
        #splitter.normalizeData()
        splitter.featureSelection()
        #splitter.encodeData()
        splitter.saveData(counter)
        counter = counter + 1



