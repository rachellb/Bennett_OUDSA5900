
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

        y_predict = (self.model.predict(self.X_test) > 0.5).astype("int32")

        specificity = specificity_score(self.Y_test, y_predict)

        gmean = geometric_mean_score(self.Y_test, y_predict)

        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        tn, fp, fn, tp = confusion_matrix(self.Y_test, y_predict).ravel()

        Results = {"Loss": score[0],
                   "Accuracy": score[1],
                   "AUC": score[4],
                   "Gmean": gmean,
                   "Recall": score[3],
                   "Precision": score[2],
                   "Specificity": specificity,
                   "True Positives": tp,
                   "True Negatives": tn,
                   "False Positives": fp,
                   "False Negatives": fn}

        return auc_, gmean

if __name__ == "__main__":

    start_time = time.time()

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
    weight = False
    model = SVMRBF(PARAMS, name='MOMI')
    X, y = model.prepData(data=path)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)
    aucList = []
    gmeanList = []
    accList = []
    precisionList = []
    recallList = []
    specList = []
    lossList = []
    historyList = []
    tpList = []
    fpList = []
    tnList = []
    fnList = []

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
        np.save('AUC/' + name + '/SVMRBFWeight_auc' + date, aucList)
        np.save('Gmean/' + name + '/SVMRBFWeight_gmean'+ date, gmeanList)
        np.save('Gmean/' + name + '/SVMRBF_gmean'+ date, accList)
        np.save('Gmean/' + name + '/SVMRBF_gmean'+ date, precisionList)
        np.save('Gmean/' + name + '/SVMRBF_gmean'+ date, recallList)
        np.save('Gmean/' + name + '/SVMRBF_gmean'+ date, gmeanList)
        np.save('Gmean/' + name + '/SVMRBF_gmean'+ date, gmeanList)


    else:
        np.save('AUC/' + name + '/SVMRBF_auc'+ date, aucList)
        np.save('Gmean/' + name + '/SVMRBF_gmean'+ date, gmeanList)

    mins = (time.time() - start_time) / 60  # Time in seconds
    print(mins)
