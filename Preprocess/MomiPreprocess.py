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

    def imputeData(self):
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

        if self.data.isnull().values.any():
            self.X_train = pd.DataFrame(np.round(MI_Imp.fit_transform(self.X_train)), columns=self.X_train.columns)
            self.X_val = pd.DataFrame(np.round(MI_Imp.transform(self.X_val)), columns=self.X_val.columns)
            self.X_test = pd.DataFrame(np.round(MI_Imp.transform(self.X_test)), columns=self.X_test.columns)

    def normalizeData(self, method):

        if method == 'MinMax':
            scaler = MinMaxScaler()
        elif method == 'StandardScale':
            scaler = StandardScaler()

        # Fit and transform training data, then transform val and test using info gained from fitting
        X_train = scaler.fit_transform(self.X_train[['MotherAge', 'MaternalHeightMeters', 'PrePregWeight', 'WeightAtAdmission',
                                                     'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions',
                                                      'HoursMembraneReptureDelivery',
                                                     'PNV_Total_Number', 'PNV_GestAge', 'PNV_Weight_Oz', 'Systolic', 'Prev_highBP']])
        X_val = scaler.transform(self.X_val[['MotherAge', 'MaternalHeightMeters', 'PrePregWeight', 'WeightAtAdmission',
                                                     'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions',
                                                     'PrePregWeight', 'WeightAtAdmission', 'HoursMembraneReptureDelivery',
                                                     'PNV_Total_Number', 'PNV_GestAge', 'PNV_Weight_Oz', 'Systolic', 'Prev_highBP']])
        X_test = scaler.transform(self.X_test[['MotherAge', 'MaternalHeightMeters', 'PrePregWeight', 'WeightAtAdmission',
                                                     'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions',
                                                     'PrePregWeight', 'WeightAtAdmission', 'HoursMembraneReptureDelivery',
                                                     'PNV_Total_Number', 'PNV_GestAge', 'PNV_Weight_Oz', 'Systolic', 'Prev_highBP']])

        X_train_imputed = pd.DataFrame(X_train, columns=self.X_train.columns)
        X_val_imputed = pd.DataFrame(X_val, columns=self.X_val.columns)
        X_test_imputed = pd.DataFrame(X_test, columns=self.X_test.columns)

        # Save newly normalized data
        self.X_train = X_train_imputed
        self.X_val = X_val_imputed
        self.X_test = X_test_imputed

    def featureSelection(self, numFeatures, method, dataset):

        self.dataset = dataset
        self.method = method

        wb = Workbook()
        wsFeatures = wb.active
        wsFeatures.title = "Features"
        filename = self.dataset + '_' + date

        # Reattach the labels - easier than saving everything separately
        self.X_train['Preeclampsia/Eclampsia'] = self.Y_train
        self.X_val['Preeclampsia/Eclampsia'] = self.Y_val
        self.X_test['Preeclampsia/Eclampsia'] = self.Y_test

        if method == 1:
            model = XGBClassifier()
            model.fit(self.X_train, self.Y_train)

            # Save graph
            ax = plot_importance(model, max_num_features=numFeatures)
            fig1 = pyplot.gcf()
            # pyplot.show()
            fig1.savefig(self.dataset + 'XGBoostTopFeatures_' + self.age + '_' + date + '.png', bbox_inches='tight')

            # Get and save best features
            feature_important = model.get_booster().get_fscore()
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
            XGBoostFeatures = list(data.index[0:numFeatures])
            XGBoostFeatures.append('Preeclampsia/Eclampsia')


            self.X_train[XGBoostFeatures].to_csv(dataset + 'XGBoost_' + self.age + '_' + date + '_train.csv', index=False)
            self.X_val[XGBoostFeatures].to_csv(dataset + 'XGBoost_' + self.age + '_' + date + '_val.csv', index=False)
            self.X_test[XGBoostFeatures].to_csv(dataset + 'XGBoost_' + self.age + '_' + date + '_test.csv', index=False)



        if method == 2:
            # instantiate SelectKBest to determine 20 best features
            fs = SelectKBest(score_func=chi2, k=numFeatures)
            fs.fit(self.X_train, self.Y_train)
            df_scores = pd.DataFrame({"Scores": fs.scores_, "P-values": fs.pvalues_})
            df_columns = pd.DataFrame(self.X_train.columns)
            # concatenate dataframes
            feature_scores = pd.concat([df_columns, df_scores], axis=1)
            feature_scores.columns = ['Feature_Name', 'Chi2 Score']  # name output columns
            feature_scores.sort_values(by=['Chi2 Score'], ascending=False, inplace=True)
            features = feature_scores.iloc[0:numFeatures]
            chi2Features = features['Feature_Name']
            chi2Features = list(chi2Features)
            chi2Features.append('Preeclampsia/Eclampsia')
            self.Chi2features = features

            for r in dataframe_to_rows(self.Chi2features, index=False, header=True):
                wsFeatures.append(r)

            wb.save(dataset + 'Chi2Features_' + self.age + '_' + date + '.xlsx')
            self.X_train[chi2Features].to_csv(dataset + 'Chi2_' + self.age + '_' + date + '_train.csv', index=False)
            self.X_val[chi2Features].to_csv(dataset + 'Chi2_' + self.age + '_' + date + '_val.csv', index=False)
            self.X_test[chi2Features].to_csv(dataset + 'Chi2_' + self.age + '_' + date + '_test.csv', index=False)

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
            wb.save(dataset + 'MIFeatures_' + self.age + '_' + date + '.xlsx')

            self.X_train[mutualInfoFeatures].to_csv(dataset + 'MI_' + self.age + '_' + date + '_train.csv', index=False)
            self.X_val[mutualInfoFeatures].to_csv(dataset + 'MI_' + self.age + '_' + date + '_val.csv', index=False)
            self.X_test[mutualInfoFeatures].to_csv(dataset + 'MI_' + self.age + '_' + date + '_test.csv', index=False)


if __name__ == "__main__":

    data = cleanDataMomi(weeks=14)
    data.splitData(testSplit=0.1, valSplit=0.1)
    data.imputeData()
    data.normalDate(method=StandardScaler)
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Processed/MOMI/WithOutliers/')
    data.featureSelection(numFeatures=20, method=1, dataset=dataPath)

