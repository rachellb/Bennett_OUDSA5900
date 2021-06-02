import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

#For feature selection
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

#For Outlier Detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import os


date = datetime.today().strftime('%m%d%y')  # For labelling purposes

class momi:

    def imputeData(self, method, data1, data2=None):
        if method == "BayesianRidge":
            estimator = BayesianRidge()
        elif method == "DecisionTree":
            estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0)
        elif method == "ExtraTrees":
            estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
        elif method == "KNN":
            estimator = KNeighborsRegressor(n_neighbors=15)

        MI_Imp = IterativeImputer(random_state=0, estimator=estimator)

        if (data2 is not None):  # If running both datasets
            if (data1.isnull().values.any() == True | data2.isnull().values.any() == True):
                data = data1.append(data2)
                self.data = pd.DataFrame(np.round(MI_Imp.fit_transform(data)), columns=data.columns)
            else:
                self.data = data1.append(data2)
        else:
            if (data1.isnull().values.any() == True):
                self.data = pd.DataFrame(np.round(MI_Imp.fit_transform(data1)), columns=data1.columns)
            else:
                self.data = data1

        return self.data

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

    def featureSelection(self, numFeatures, method, dataset):

        self.dataset=dataset
        self.method = method

        wb = Workbook()
        wsFeatures = wb.active
        wsFeatures.title = "Features"
        filename = self.dataset + '_' + date

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

            self.data[XGBoostFeatures].to_csv(dataset + 'XGBoost_' + self.age +'_' + date + '.csv', index=False)

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

            wb.save(dataset +'Chi2Features_' + self.age + '_' + date + '.xlsx')
            self.data[chi2Features].to_csv(dataset + 'Chi2_' + self.age +'_' + date + '.csv', index=False)

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
            wb.save(dataset +'MIFeatures_' + self.age + '_' + date + '.xlsx')

            self.data[mutualInfoFeatures].to_csv(dataset + 'MI_' + self.age + '_' + date +'.csv', index=False)