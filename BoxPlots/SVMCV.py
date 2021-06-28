
import pandas as pd
import numpy as np
import time
import os

#For model
from sklearn.svm import SVC

# For stratified Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold

# For Auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from Preprocess import preProcess

class SVMRBF(preProcess):

    def prepData(self, data):

        self.data = pd.read_csv(data)

        X = self.data.drop(columns='Preeclampsia/Eclampsia')
        Y = self.data['Preeclampsia/Eclampsia']

        return X, Y

    def setData(self, X_train, X_test, Y_train, Y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test.ravel()

    def buildModel(self, weight = False):

        if weight:
            clfSG = SVC(class_weight='balanced', verbose=1)
        else:
            clfSG = SVC(verbose=1)

        clfSG.fit(self.X_train, self.Y_train)

        self.predictions = clfSG.predict(self.X_test).ravel()

    def evaluateModel(self):

        fpr, tpr, thresholds = roc_curve(self.Y_test, self.predictions)
        auc_ = metrics.auc(fpr, tpr)
        gmean = geometric_mean_score(self.Y_test, self.predictions)

        return auc_, gmean

if __name__ == "__main__":

    start_time = time.time()

    PARAMS = {'estimator': "BayesianRidge",
              'Normalize': 'MinMax',
              'OutlierRemove': 'lof',
              'Feature_Selection': 'Chi2',
              'Feature_Num': 30,
              'TestSplit': 0.10,
              'ValSplit': 0.10,
              'dataset': 'MOMI'}

    # Get path to cleaned data
    parent = os.path.dirname(os.getcwd())
    path = os.path.join(parent, 'Data/Processed/MOMI/WithOutliers/momiUSPre_062621.csv')

    name = 'MOMI'
    weight = True
    model = SVMRBF(PARAMS, name='MOMI')
    X, y = model.prepData(data=path)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)
    aucList = []
    gmeanList = []

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.setData(X_train, X_test, y_train, y_test)
        model.imputeData()
        model.detectOutliers()
        model.normalizeData()
        model.featureSelection()
        model.encodeData()
        model.buildModel(weight=weight)
        auc, gmean = model.evaluateModel()

        aucList.append(auc)
        gmeanList.append(gmean)

    if weight:
        np.save('AUC/' + name + '/SVMRBFWeight_auc', aucList)
        np.save('Gmean/' + name + '/SVMRBFWeight_gmean', gmeanList)
    else:
        np.save('AUC/' + name + '/SVMRBF_auc', aucList)
        np.save('Gmean/' + name + '/SVMRBF_gmean', gmeanList)

    mins = (time.time() - start_time) / 60  # Time in seconds
    print(mins)
