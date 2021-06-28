
import pandas as pd
import numpy as np
import time
import os

#For model
from sklearn.linear_model import SGDClassifier

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

    def prepTuningData(self, splitCoeff=0.1):  # Why is it split this way?
        # Splitting the training set according to split coefficient
        splitRow = int(splitCoeff * len(self.X_train))
        self.X_tuneTrain = self.X_train[0:splitRow]
        self.X_tuneTest = self.X_train[splitRow:len(self.X_train)]
        self.y_tuneTrain = self.Y_train[0:splitRow]
        self.y_tuneTest = self.Y_train[splitRow:len(self.Y_train)]

    def tuneSVM(self, weight):

        self.prepTuningData()
        # The following finds the best alpha for which we get the highest AUC
        alphaSet = np.array([0.01, 0.1, 1, 10, 100, 1000])
        self.bestAlpha = alphaSet[0]
        bestAUC = 0
        for alpha in alphaSet:
            auc, gmean = self.runSVM(alpha, weight)
            if auc > bestAUC:
                self.bestAlpha = alpha
                bestAUC = auc
                # print(bestAUC)
                # print(self.bestAlpha)


    def runSVM(self, alpha, weight=False):

        if weight:
            clfSG = SGDClassifier(loss="hinge", max_iter=1000, alpha=alpha, tol=1e-3, class_weight='balanced')
        else:
            clfSG = SGDClassifier(loss="hinge", max_iter=1000, alpha=alpha, tol=1e-3)

        clfSG.fit(self.X_train, self.Y_train)

        self.predictions = clfSG.predict(self.X_test).ravel()
        fpr, tpr, thresholds = roc_curve(self.Y_test, self.predictions)
        auc_ = metrics.auc(fpr, tpr)
        gmean = geometric_mean_score(self.Y_test, self.predictions)

        return auc_, gmean

    def classify(self, weight):

        auc, gmean = self.runSVM(alpha=self.bestAlpha, weight=weight)

        return auc, gmean

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
        model.tuneSVM(weight)
        auc, gmean = model.classify(weight)


        aucList.append(auc)
        gmeanList.append(gmean)


    if weight:
        np.save('AUC/' + name + '/SVMLinWeight_auc', aucList)
        np.save('Gmean/' + name + '/SVMLinWeight_gmean', gmeanList)

    else:
        np.save('AUC/' + name + '/SVMLin_auc', aucList)
        np.save('Gmean/' + name + '/SVMLin_gmean', gmeanList)

    mins = (time.time() - start_time) / 60  # Time in seconds
    print(mins)
