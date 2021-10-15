
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

class SVMRBF():

    def setData(self, X_train, Y_train, X_test):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def buildModel(self, weight = False):

        if weight:
            clfSG = SVC(class_weight='balanced', verbose=1)
        else:
            clfSG = SVC(verbose=1)

        clfSG.fit(self.X_train, self.Y_train)

        #self.predictions = clfSG.predict(self.X_test).ravel()

        self.predictions = (clfSG.predict(self.X_test) > 0.5).astype("int32")
        preds = pd.DataFrame(self.predictions)
        return preds

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




    name = 'MOMI'
    weight = False
    model = SVMRBF()

    counter =1
    while (counter <= 50):

        X_train = pd.read_csv('Data/' + name + '/092021_X_Train_' + str(counter) + '.csv')
        Y_train = pd.read_csv('Data/' + name + '/092021_Y_Train_' + str(counter) + '.csv')
        X_test = pd.read_csv('Data/' + name + '/092021_X_Test_' + str(counter) + '.csv')

        model.setData(X_train, Y_train, X_test)
        predictions = model.buildModel(weight=weight)

        if weight:
            predictions.to_csv('Predictions/' + name + '/SVMRBF/Weighted/CV_' + str(counter) + '.csv', index=False,
                               header=False)
        else:
            predictions.to_csv('Predictions/' + name + '/SVMRBF/Unweighted/CV_' + str(counter) + '.csv', index=False,
                               header=False)

        counter = counter + 1


