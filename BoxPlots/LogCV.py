
import pandas as pd
import numpy as np
import time
import os

#For model
from sklearn.linear_model import LogisticRegression

# For stratified Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold

# For Auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from imblearn.metrics import geometric_mean_score

from Preprocess import preProcess


class LogReg(preProcess):

    def prepData(self, data):

        self.data = pd.read_csv(data)

        X = self.data.drop(columns='Preeclampsia/Eclampsia')
        Y = self.data['Preeclampsia/Eclampsia']

        return X, Y


    def buildModel(self, weight = False):

        if not weight:
            logReg = LogisticRegression()
        else:
            logReg = LogisticRegression(class_weight='balanced')

        logReg.fit(self.X_train, self.Y_train)

        self.predictions = logReg.predict(self.X_test).ravel()



    def evaluateModel(self):

        fpr, tpr, thresholds = roc_curve(self.Y_test, self.predictions)
        auc_ = metrics.auc(fpr, tpr)
        gmean = geometric_mean_score(self.Y_test, self.predictions)

        return auc_, gmean


if __name__ == "__main__":

    PARAMS = {'estimator': "BayesianRidge",
              'Normalize': 'MinMax',
              'OutlierRemove': 'None',
              'Feature_Selection': 'Chi2',
              'Feature_Num': 1000,
              'TestSplit': 0.10,
              'ValSplit': 0.10,
              'dataset': 'MOMI'}

    # Get path to cleaned data
    parent = os.path.dirname(os.getcwd())
    path = os.path.join(parent, 'Preprocess/momiEncoded_061521.csv')

    name = 'MOMI'
    weight = True
    model = LogReg(PARAMS, name='MOMI')
    X, y = model.prepData(data=path)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)

    aucList = []
    gmeanList = []

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.setData(X_train, X_test, y_train, y_test)
        model.imputeData()
        #model.detectOutliers()
        model.normalizeData()
        model.featureSelection()
        model.encodeData()
        model.buildModel(weight=weight)

        auc, gmean = model.evaluateModel()
        aucList.append(auc)
        gmeanList.append(gmean)


    if weight:
        np.save('AUC/' + name + '/logWeight_auc', aucList)
        np.save('Gmean/' + name + '/logWeight_gmean', gmeanList)
    else:
        np.save('AUC/' + name + '/log_auc', aucList)
        np.save('Gmean/' + name + '/log_gmean', gmeanList)

