
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

class SVMLin():

    def setData(self, X_train, Y_train, X_test):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def prepTuningData(self, splitCoeff=0.1):  # Why is it split this way?
        # Splitting the training set according to split coefficient
        splitRow = int(splitCoeff * len(self.X_train))
        X_tuneTrain = self.X_train[0:splitRow]
        X_tuneTest = self.X_train[splitRow:len(self.X_train)]
        y_tuneTrain = self.Y_train[0:splitRow]
        y_tuneTest = self.Y_train[splitRow:len(self.Y_train)]

        return X_tuneTrain, y_tuneTrain, X_tuneTest, y_tuneTest

    def tuneSVM(self, weight):

        X_tuneTrain, y_tuneTrain, X_tuneTest, y_tuneTest = self.prepTuningData()

        # The following finds the best alpha for which we get the highest AUC
        alphaSet = np.array([0.01, 0.1, 1, 10, 100, 1000])
        self.bestAlpha = alphaSet[0]
        bestAUC = 0
        for alpha in alphaSet:
            auc, gmean = self.runSVM(alpha, X_tuneTrain, y_tuneTrain, X_tuneTest, y_tuneTest, weight)
            if auc > bestAUC:
                self.bestAlpha = alpha
                bestAUC = auc
                # print(bestAUC)
                # print(self.bestAlpha)


    def runSVM(self, alpha, X_train, Y_Train, X_Test, Y_test=None, weight=False):

        if (Y_test is not None):
            if weight:
                clfSG = SGDClassifier(loss="hinge", max_iter=1000, alpha=alpha, tol=1e-3, class_weight='balanced')
            else:
                clfSG = SGDClassifier(loss="hinge", max_iter=1000, alpha=alpha, tol=1e-3)

            clfSG.fit(X_train, Y_Train)

            predictions = clfSG.predict(X_Test).ravel()
            unique = np.unique(predictions)
            uniqueY = np.unique(Y_test)
            fpr, tpr, thresholds = roc_curve(Y_test, predictions)
            auc_ = metrics.auc(fpr, tpr)
            gmean = geometric_mean_score(Y_test, predictions)

            return auc_, gmean

        else:
            if weight:
                clfSG = SGDClassifier(loss="hinge", max_iter=1000, alpha=alpha, tol=1e-3, class_weight='balanced')
            else:
                clfSG = SGDClassifier(loss="hinge", max_iter=1000, alpha=alpha, tol=1e-3)

            clfSG.fit(X_train, Y_train)
            self.predictions = (clfSG.predict(self.X_test) > 0.5).astype("int32")
            preds = pd.DataFrame(self.predictions)

            return preds

    def classify(self, weight):

        preds = self.runSVM(X_train=self.X_train, Y_Train=self.Y_train, X_Test=self.X_test, alpha=self.bestAlpha,
                            weight=weight)
        return preds

if __name__ == "__main__":

    name = 'MOMI'
    weight = False

    model = SVMLin()
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)

    predsList = []
    counter = 1
    while (counter <= 50):

        X_train = pd.read_csv('Data/' + name + '/092621_X_Train_' + str(counter) + '.csv')
        Y_train = pd.read_csv('Data/' + name + '/092621_Y_Train_' + str(counter) + '.csv')
        X_test = pd.read_csv('Data/' + name + '/092621_X_Test_' + str(counter) + '.csv')

        model.setData(X_train, Y_train, X_test)
        model.tuneSVM(weight)
        predictions = model.classify(weight)

        if weight:
            predictions.to_csv('Predictions/' + name + '/SVMLin/Weighted/CV_' + str(counter) + '.csv', index=False, header=False)
        else:
            predictions.to_csv('Predictions/' + name + '/SVMLin/Unweighted/CV_' + str(counter) + '.csv', index=False, header=False)

        counter = counter + 1