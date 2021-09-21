import pandas as pd
import numpy as np

# For additional metrics
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def evaluateModel(Y_test, predictions):

    specificity = specificity_score(Y_test, predictions)
    gmean = geometric_mean_score(Y_test, predictions)
    recall = metrics.recall_score(Y_test, predictions)
    precision = metrics.precision_score(Y_test, predictions)
    accuracy = metrics.accuracy_score(Y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()

    Results = {#"Loss": score[0],
               "Accuracy": accuracy,
               #"AUC": auc_,
               #"Gmean": gmean,
               "Recall": recall,
               "Precision": precision,
               "Specificity": specificity}
               #"True Positives": tp,
               #"True Negatives": tn,
               #"False Positives": fp,
               #"False Negatives": fn}

    return Results

if __name__ == "__main__":

    name = 'Oklahoma'
    model = 'LR'
    weight = True

    counter = 1

    #aucList = []
    #gmeanList = []
    accList = []
    precisionList = []
    recallList = []
    specList = []

    while (counter <= 50):

        if (weight):
            predictions = pd.read_csv('Predictions/' + name + '/' + model +'/Weighted' + '/CV_' + str(counter) + '.csv', header=None)
        else:
            predictions = pd.read_csv(
                'Predictions/' + name + '/' + model + '/Unweighted' + '/CV_' + str(counter) + '.csv', header=None)

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
        np.save('Results/ACC/' + name + '/' + model + 'Weight_acc', accList)
        np.save('Results/PR/' + name + '/' + model + 'Weight_pr', precisionList)
        np.save('Results/RE/' + name + '/' + model + 'Weight_re', recallList)
        np.save('Results/SP/' + name + '/' + model + 'Weight_sp', specList)

    else:
        np.save('Results/ACC/' + name + '/' + model + 'Unweight_acc', accList)
        np.save('Results/PR/' + name + '/' + model + 'Unweight_pr', precisionList)
        np.save('Results/RE/' + name + '/' + model + 'Unweight_re', recallList)
        np.save('Results/SP/' + name + '/' + model + 'Unweight_sp', specList)

