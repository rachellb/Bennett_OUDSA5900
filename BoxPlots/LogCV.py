
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

# For additional metrics
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import confusion_matrix

from Preprocess import preProcess



class LogReg(preProcess):

    def setData(self, X_train, X_test):

        self.X_train = X_train
        self.X_test = X_test

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
        return self.predictions

    def evaluateModel(self):

        fpr, tpr, thresholds = roc_curve(self.Y_test, self.predictions)
        auc_ = metrics.auc(fpr, tpr)
        gmean = geometric_mean_score(self.Y_test, self.predictions)

        specificity = specificity_score(self.Y_test, self.predictions)

        gmean = geometric_mean_score(self.Y_test, self.predictions)

        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        tn, fp, fn, tp = confusion_matrix(self.Y_test, self.predictions).ravel()

        Results = {"Loss": score[0],
                   "Accuracy": score[1],
                   "AUC": auc_,
                   "Gmean": gmean,
                   "Recall": score[3],
                   "Precision": score[2],
                   "Specificity": specificity,
                   "True Positives": tp,
                   "True Negatives": tn,
                   "False Positives": fp,
                   "False Negatives": fn}

        return Results


if __name__ == "__main__":

    PARAMS = {'estimator': "BayesianRidge",
              'Normalize': 'MinMax',
              'OutlierRemove': 'None',
              'Feature_Selection': 'Chi2',
              'Feature_Num': 1000,
              'TestSplit': 0.10,
              'ValSplit': 0.10,
              'dataset': 'MOMI'}

    name = 'MOMI'
    weight = True
    model = LogReg(PARAMS, name='MOMI')

    predsList = []
    counter = 1
    parent = os.path.dirname(os.getcwd())

    for i in range(10):

        X_train = pd.read_csv('Data/' + name + '/092021_X_Train_' + str(counter) + '.csv')
        X_test = pd.read_csv('Data/' + name + '/092021_X_Train_' + str(counter) + '.csv')

        model.setData(X_train, X_test)
        predictions = model.buildModel(weight=weight)
        predsList.append(predictions)

    allPreds= pd.DataFrame(predsList)

    if weight:
        allPreds.to_csv('Predictions/' + name + 'logWeight.csv', index=False, header=False)
    else:
        allPreds.to_csv('Predictions/' + name + 'log.csv', index=False, header=False)

    counter = counter + 1
