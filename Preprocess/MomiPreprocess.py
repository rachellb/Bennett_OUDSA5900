import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

# For feature selection
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from datetime import datetime
from openpyxl import Workbook  # For storing results
from openpyxl.utils.dataframe import dataframe_to_rows
import openpyxl

# For Outlier Detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import os
from Cleaning.Clean import *

date = datetime.today().strftime('%m%d%y')  # For labelling purposes


def encodeCols(data):
    encodeCols = ['Insurance', 'OutcomeOfDelivery','MaternalNeuromuscularDisease','MCollagenVascularDisease',
                  'MStructuralHeartDiseas', 'MPostPartumComplications','DiabetesMellitus', 'ThyroidDisease',
                  'MLiverGallPanc','KidneyDisease', 'MAnemiaWOHemoglobinopathy',  'MHemoglobinopathy',
                  'Thrombocytopenia','ViralOrProtoInf','OtherSubstanceAbuse', 'InfSex',  'CNSAbnormality',
                  'RaceCollapsed']

    for column in data.columns:
        if column in encodeCols:
            data = pd.get_dummies(data, columns=[column])

    return data


class momi:

    def __init__(self, data):

        self.data = data

    def splitData(self, testSize, valSize):

        X = self.data.drop(columns='Preeclampsia/Eclampsia')
        Y = self.data['Preeclampsia/Eclampsia']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, stratify=Y,
                                                                                test_size=testSize,
                                                                                random_state=7)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              stratify=self.Y_train,
                                                                              test_size=valSize,
                                                                              random_state=8)

    def imputeData(self, method):
        # Scikitlearn's Iterative imputer
        # Default imputing method is Bayesian Ridge Regression

        if method == "BayesianRidge":
            estimator = BayesianRidge()
        elif method == "DecisionTree":
            estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0)
        elif method == "ExtraTrees":
            estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
        elif method == "KNN":
            estimator = KNeighborsRegressor(n_neighbors=15)

        MI_Imp = IterativeImputer(random_state=0, estimator=estimator)


        if self.data.isnull().values.any():
            self.X_train_imputed = pd.DataFrame(MI_Imp.fit_transform(self.X_train), columns=self.X_train.columns)
            self.X_val_imputed = pd.DataFrame(MI_Imp.transform(self.X_val), columns=self.X_val.columns)
            self.X_test_imputed = pd.DataFrame(MI_Imp.transform(self.X_test), columns=self.X_test.columns)

            # Rounding only the categorical variables that were imputed
            self.X_train = self.X_train_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
                                                       'DeliveriesPriorAdmission': 0, 'TotalAbortions': 0,
                                                       'Primagrivada': 0, 'MaternalNeuromuscularDisease': 0,
                                                       'KidneyDisease': 0, 'Thrombocytopenia': 0, 'InfSex': 0,
                                                       'CNSAbnormality': 0, 'CongenitalSyphilis': 0, 'UTI': 0,
                                                       'RaceCollapsed': 0, 'Systolic': 0})
            self.X_val = self.X_val_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
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
        self.X_val['RaceCollapsed'] = np.where(((self.X_val['RaceCollapsed'] > 4)), 4,
                                               self.X_val['RaceCollapsed'])
        self.X_test['RaceCollapsed'] = np.where(((self.X_test['RaceCollapsed'] > 4)), 4,
                                                self.X_test['RaceCollapsed'])

        # Fix incorrectly imputed value
        self.X_train['RaceCollapsed'] = np.where(((self.X_train['RaceCollapsed'] < 0)), 0,
                                                 self.X_train['RaceCollapsed'])
        self.X_val['RaceCollapsed'] = np.where(((self.X_val['RaceCollapsed'] < 0)), 0,
                                               self.X_val['RaceCollapsed'])
        self.X_test['RaceCollapsed'] = np.where(((self.X_test['RaceCollapsed'] < 0)), 0,
                                                self.X_test['RaceCollapsed'])

    def normalizeData(self, method):

        if method == 'MinMax':
            scaler = MinMaxScaler()
        elif method == 'StandardScale':
            scaler = StandardScaler()

        # Fit and transform training data, then transform val and test using info gained from fitting
        scaleColumns = ['MotherAge', 'WeightAtAdmission',
                        'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions', 'WeightAtAdmission',
                        'PNV_GestAge', 'PNV_Weight_Oz', 'Systolic', 'Prev_highBP']

        self.X_train[scaleColumns] = scaler.fit_transform(self.X_train[scaleColumns])
        self.X_val[scaleColumns] = scaler.fit_transform(self.X_val[scaleColumns])
        self.X_test[scaleColumns] = scaler.fit_transform(self.X_test[scaleColumns])

    def featureSelection(self, numFeatures, method, dataset):

        self.dataset = dataset
        self.method = method

        wb = Workbook()
        wsFeatures = wb.active
        wsFeatures.title = "Features"
        filename = self.dataset + '_' + date


        # One hot encoding
        self.X_train = encodeCols(self.X_train)
        self.X_val = encodeCols(self.X_val)
        self.X_test = encodeCols(self.X_test)

        if method == 1:
            model = XGBClassifier()
            model.fit(self.X_train, self.Y_train)

            # Save graph
            ax = plot_importance(model, max_num_features=numFeatures)
            fig1 = pyplot.gcf()
            # pyplot.show()
            fig1.savefig(self.dataset + 'XGBoostTopFeatures_' + date + '.png', bbox_inches='tight')

            # Get and save best features
            feature_important = model.get_booster().get_fscore()
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
            XGBoostFeatures = list(data.index[0:numFeatures])
            XGBoostFeatures.append('Preeclampsia/Eclampsia')

            self.X_train = self.X_train.set_index(self.Y_train.index)
            self.X_val = self.X_val.set_index(self.Y_val.index)
            self.X_test = self.X_test.set_index(self.Y_test.index)

            # Reattach the labels - easier than saving everything separately
            self.X_train['Preeclampsia/Eclampsia'] = self.Y_train
            self.X_val['Preeclampsia/Eclampsia'] = self.Y_val
            self.X_test['Preeclampsia/Eclampsia'] = self.Y_test

            self.X_train = self.X_train[XGBoostFeatures]
            self.X_val = self.X_val[XGBoostFeatures]
            self.X_test = self.X_test[XGBoostFeatures]

            self.X_train.to_csv(dataset + 'XGBoost_' + date + '_train.csv', index=False)
            self.X_val.to_csv(dataset + 'XGBoost_' + date + '_val.csv', index=False)
            self.X_test.to_csv(dataset + 'XGBoost_' + date + '_test.csv', index=False)

        if method == 2:
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
            chi2Features = list(chi2Features)
            chi2Features.append('Preeclampsia/Eclampsia')
            self.Chi2features = features

            for r in dataframe_to_rows(self.Chi2features, index=False, header=True):
                wsFeatures.append(r)

            self.X_train = self.X_train.set_index(self.Y_train.index)
            self.X_val = self.X_val.set_index(self.Y_val.index)
            self.X_test = self.X_test.set_index(self.Y_test.index)

            # Reattach the labels - easier than saving everything separately
            self.X_train['Preeclampsia/Eclampsia'] = self.Y_train
            self.X_val['Preeclampsia/Eclampsia'] = self.Y_val
            self.X_test['Preeclampsia/Eclampsia'] = self.Y_test


            wb.save(dataset + 'Chi2Features_' + date + '.xlsx')
            self.X_train.to_csv(dataset + 'Chi2_' + date + '_train.csv', index=False)
            self.X_val.to_csv(dataset + 'Chi2_' + date + '_val.csv', index=False)
            self.X_test.to_csv(dataset + 'Chi2_' + date + '_test.csv', index=False)

        if method == 3:
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
            mutualInfoFeatures = list(mutualInfoFeatures)
            mutualInfoFeatures.append('Preeclampsia/Eclampsia')
            self.MIFeatures = features

            for r in dataframe_to_rows(self.MIFeatures, index=False, header=True):
                wsFeatures.append(r)
            wb.save(dataset + 'MIFeatures_' + date + '.xlsx')

            self.X_train = self.X_train.set_index(self.Y_train.index)
            self.X_val = self.X_val.set_index(self.Y_val.index)
            self.X_test = self.X_test.set_index(self.Y_test.index)

            # Reattach the labels - easier than saving everything separately
            self.X_train['Preeclampsia/Eclampsia'] = self.Y_train
            self.X_val['Preeclampsia/Eclampsia'] = self.Y_val
            self.X_test['Preeclampsia/Eclampsia'] = self.Y_test

            # Select only the top 20 features plus label
            self.X_train = self.X_train[mutualInfoFeatures]
            self.X_val = self.X_val[mutualInfoFeatures]
            self.X_test = self.X_test[mutualInfoFeatures]

            self.X_train.to_csv(dataset + 'MI_' + date + '_train.csv', index=False)
            self.X_val.to_csv(dataset + 'MI_' + date + '_val.csv', index=False)
            self.X_test.to_csv(dataset + 'MI_' + date + '_test.csv', index=False)


if __name__ == "__main__":

    data = cleanDataMomi(weeks=14)
    #data = pd.read_csv('momiEncoded_Full_060821.csv')
    preProcess = momi(data)
    preProcess.splitData(testSize=0.1, valSize=0.1)
    preProcess.imputeData(method="BayesianRidge")
    preProcess.normalizeData(method="MinMax")

    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Processed/MOMI/WithOutliers/oneHot/')
    preProcess.featureSelection(numFeatures=20, method=3, dataset=dataPath)
