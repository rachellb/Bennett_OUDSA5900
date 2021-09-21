import pandas as pd
import numpy as np

# For additional metrics
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import confusion_matrix



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

    name = 'Oklahoma'
    model = 'LR'
    weight = False

    counter = 1

    #aucList = []
    #gmeanList = []
    accList = []
    precisionList = []
    recallList = []
    specList = []

    while (counter <= 50):

        if (weight):
            predictions = pd.read_csv('Data/' + name + '/' + model +'/Weighted' + '/092121_X_Train_' + str(counter) + '.csv')
        else:
            predictions = pd.read_csv(
                'Data/' + name + '/' + model + '/Unweighted' + '/092121_X_Train_' + str(counter) + '.csv')

        Y_values = pd.read_csv('Data/' + name + '/092121_Y_Test_' + str(counter) + '.csv')

        Results = evaluateModel(Y_values, predictions)

        #aucList.append(Results["AUC"])
        #gmeanList.append(Results["Gmean"])
        accList.append(Results["Accuracy"])
        precisionList.append(Results["Precision"])
        recallList.append(Results["Recall"])
        specList.append(Results["Specificity"])

        counter = counter + 1

    if (weight):
        np.save('ACC/' + name + '/' + model + 'Weight_acc', accList)
        np.save('PR/' + name + '/' + model + 'Weight_pr', precisionList)
        np.save('RE/' + name + '/' + model + 'Weight_re', recallList)
        np.save('SP/' + name + '/' + model + 'Weight_sp', specList)

    else:
        np.save('ACC/' + name + '/' + model + 'Unweight_acc', accList)
        np.save('PR/' + name + '/' + model + 'Unweight_pr', precisionList)
        np.save('RE/' + name + '/' + model + 'Unweight_re', recallList)
        np.save('SP/' + name + '/' + model + 'Unweight_sp', specList)

