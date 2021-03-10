
import pandas as pd
import numpy as np
import time
import os

#For model
from sklearn.linear_model import LogisticRegression

# For stratified Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold

# For Auc
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from imblearn.metrics import geometric_mean_score

class LogReg():

    def prepData(self, data):

        data = pd.read_csv(data)

        X = data.drop(columns='Preeclampsia/Eclampsia')
        Y = data['Preeclampsia/Eclampsia']

        return X, Y

    def setData(self, X_train, X_test, Y_train, Y_test):

        self.X_train = X_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.Y_train = Y_train.to_numpy()
        self.Y_test = Y_test.to_numpy().ravel()


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

    parent = os.path.dirname(os.getcwd())

    pathOK = os.path.join(parent, 'Data/Processed/Oklahoma/Complete/Full/Outliers/Chi2_Categorical.csv')
    pathTX = os.path.join(parent, 'Data/Processed/Texas/Full/Outliers/Complete/Chi2_Categorical.csv')

    name = 'Oklahoma'
    weight = False
    model = LogReg()
    X, y = model.prepData(data=pathOK)

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)

    aucList = []
    gmeanList = []

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.setData(X_train, X_test, y_train, y_test)

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

