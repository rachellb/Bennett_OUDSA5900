import numpy as np
import pandas as pd

from sklearn.svm import SVC


class SVM():

    def __init__(self, weight, data):
        self.weight = weight
        self.data = data

    def prepData(self, data):
        path = data

        self.X_train = pd.read_csv(path + 'X_Train.csv')
        self.Y_train = pd.read_csv(path + 'Y_Train.csv')
        self.X_test = pd.read_csv(path + 'X_Test.csv')
        self.Y_test = pd.read_csv(path + 'Y_Test.csv')
        self.X_val = pd.read_csv(path + 'X_Val.csv')
        self.Y_val = pd.read_csv(path + 'Y_Val.csv')

        # Set all to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy().ravel()
        self.X_val = self.X_val.to_numpy()
        self.Y_val = self.Y_val.to_numpy().ravel()
        self.X_test = self.X_test.to_numpy()
        self.Y_test = self.Y_test.to_numpy().ravel()

    def prepTuningData(self, splitCoeff=0.1):  # Why is it split this way?
        # Splitting the training set according to split coefficient
        splitRow = int(splitCoeff * len(self.X_train))
        self.X_tuneTrain = self.X_train[0:splitRow]
        self.X_tuneTest = self.X_train[splitRow:len(self.X_train)]
        self.y_tuneTrain = self.Y_train[0:splitRow]
        self.y_tuneTest = self.Y_train[splitRow:len(self.Y_train)]

    def runSVM(self):

        if self.weight:
            clfSG = SVC(class_weight='balanced', verbose=1)
        else:
            clfSG = SVC(verbose=1)

        clfSG.fit(self.X_train, self.Y_train)

        y_predSG = clfSG.predict(self.X_test).ravel()

        return y_predSG

    def tuneSVM(self):

        self.prepTuningData()
        # The following finds the best alpha for which we get the highest AUC
        alphaSet = np.array([0.01, 0.1, 1, 10, 100, 1000])
        self.bestAlpha = alphaSet[0]
        bestAUC = 0
        for alpha in alphaSet:
            output = self.runSVM(alpha)
            if output[3] > bestAUC:
                self.bestAlpha = alpha
                bestAUC = output[3]
                # print(bestAUC)
                # print(self.bestAlpha)

    def classify(self):

        y_predSG = self.runSVM()

        if self.weight:
            np.save('Predictions/' + self.data + '/svmRBFWeight_pred', y_predSG)
        else:
            np.save('Predictions/' + self.data + '/svmRBF_pred', y_predSG)

if __name__ == "__main__":
    data = 'Oklahoma'

    model = SVM(weight=True, data=data)
    model.prepData(data='Data/' + data + '/')
    model.classify()

