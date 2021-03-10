
import pandas as pd
import numpy as np
import time

#For model
from sklearn.linear_model import LogisticRegression

#For getting usable output
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class LogReg():

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


    def buildModel(self, data, weight = False):

        if not weight:
            logReg = LogisticRegression()
        else:
            logReg = LogisticRegression(class_weight='balanced')

        logReg.fit(self.X_train, self.Y_train)

        y_predLog = logReg.predict(self.X_test).ravel()

        if weight:
            np.save('Predictions/' + data + '/logWeight_pred', y_predLog)
        else:
            np.save('Predictions/' + data + '/log_pred', y_predLog)

if __name__ == "__main__":
    data = 'Oklahoma'

    logmodel = LogReg()
    logmodel.prepData(data='Data/' + data + '/')
    logmodel.buildModel(weight=True, data=data)


