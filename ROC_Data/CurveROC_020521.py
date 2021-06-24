import numpy as np
import pandas as pd

# For Drawing curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# For Record Keeping
from datetime import datetime
date = datetime.today().strftime('%m%d%y')  # For labelling purposes

class Curve():

    def __init__(self, data):
        self.data = data

    def loadData(self):

        self.logpred = np.load('Predictions/'+self.data+ '/log_pred.npy')
        self.logWeight = np.load('Predictions/'+self.data+ '/logWeight_pred.npy')

        self.svmLin = np.load('Predictions/'+self.data+ '/svmLin_pred.npy')
        self.svmLinWeight = np.load('Predictions/'+self.data+ '/svmLinWeight_pred.npy')

        self.svmRBF = np.load('Predictions/'+self.data+ '/svmRBF_pred.npy')
        self.svmRBFWeight = np.load('Predictions/'+self.data+ '/svmRBFWeight_pred.npy')

        self.nn = np.load('Predictions/'+self.data+ '/nn_pred.npy')
        self.nnWeight = np.load('Predictions/'+self.data+ '/nnWeight_pred.npy')

        self.nn = np.load('Predictions/' + self.data + '/nn_pred.npy')
        self.nnWeight = np.load('Predictions/' + self.data + '/nnWeight_pred.npy')

        self.predictions = {"LR": self.logpred,
                            "WLR": self.logWeight,
                            "SVM-Lin": self.svmLin,
                            "WSVM-Lin": self.svmLinWeight,
                            "SVM-RBF": self.svmRBF,
                            "WSVM-RBF": self.svmRBFWeight,
                            "DNN": self.nn,
                            "CSDNN": self.nnWeight}

        filename = 'Data/'+ self.data + '/Y_Test.csv'

        self.Y_test = pd.read_csv(filename).to_numpy().ravel()


    def getRate(self):

        # A dictionary that will hold the false positve, true positive, and threshold rate for each model
        self.params = {}

        linestyle = [':','-','']

        for model, predictions in self.predictions.items():
            fpr, tpr, thresholds = roc_curve(self.Y_test, predictions)
            auc_ = auc(fpr, tpr)

            self.params[model] = {"tpr": tpr,
                                  "fpr": fpr,
                                  "thresholds": thresholds,
                                  "auc": auc_}

    def graph_rates(self):
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')


        """
        for model in self.params.keys():
            plt.plot(self.params[model]["fpr"], self.params[model]["tpr"], label=model)
        """


        plt.plot(self.params["LR"]["fpr"], self.params["LR"]["tpr"], label="LR", linestyle='-.', color='goldenrod', linewidth=2)
        plt.plot(self.params["WLR"]["fpr"], self.params["WLR"]["tpr"], label="WLR", linestyle=':', color='cadetblue')
        plt.plot(self.params["SVM-Lin"]["fpr"], self.params["SVM-Lin"]["tpr"], label="SVM-Lin", linestyle='--', color='purple', linewidth=2)
        plt.plot(self.params["WSVM-Lin"]["fpr"], self.params["WSVM-Lin"]["tpr"], label="WSVM-Lin", linestyle='--', color='maroon')
        plt.plot(self.params["SVM-RBF"]["fpr"], self.params["SVM-RBF"]["tpr"], label="SVM-RBF", linestyle='solid', color='green', alpha=0.4, linewidth=3)
        plt.plot(self.params["WSVM-RBF"]["fpr"], self.params["WSVM-RBF"]["tpr"], label="WSVM-RBF", linestyle='-.', color='coral')
        plt.plot(self.params["DNN"]["fpr"], self.params["DNN"]["tpr"], label="DNN", linestyle='solid', alpha=0.5, color='blue', linewidth=3)
        plt.plot(self.params["CSDNN"]["fpr"], self.params["CSDNN"]["tpr"], label="CSDNN", linestyle=':', color='black',linewidth=3)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(self.data + ' ROC curve')
        plt.legend(loc='lower right')
        plt.savefig('Graphs/Animation/' + self.data+ "_ROC_Curve16_" + date, bbox_inches="tight")
        plt.show()



if __name__ == "__main__":

    models = Curve(data='Oklahoma')
    models.loadData()
    models.getRate()
    models.graph_rates()



