#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For handling data
import pandas as pd
import numpy as np
from datetime import datetime
import os

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.utils import class_weight
from statistics import mean

# For imputing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#For Outlier Detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# For feature selection
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.feature_selection import mutual_info_classif

# For balancing batches
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler

# For NN and tuning
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# For additional metrics
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import confusion_matrix
import tensorflow_addons as tfa  # For focal loss function
import time
import matplotlib.pyplot as plt
import statistics
from PIL import Image


# For Stratified Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold

date = datetime.today().strftime('%m%d%y')  # For labelling purposes

# For recording results
import neptune
from neptunecontrib.api.table import log_table
from neptunecontrib.monitoring.keras import NeptuneMonitor
from secret import api_
from Cleaning.Clean import *

# Initialize the project
neptune.init(project_qualified_name='rachellb/Comparisons', api_token=api_)







def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):

    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=tf.float64)
        tf_y_pred = tf.cast(y_pred, dtype=tf.float64)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])

        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = K.mean(tf.math.multiply(ce, weights_v))

        return loss

    return weighted_cross_entropy_fn


class fullNN():

    def __init__(self, PARAMS):

        self.PARAMS = PARAMS

    def prepData(self, age, data=None, path=None):

        self.age = age

        if path is not None:
            self.data = pd.read_csv(path)
            X = data.drop(columns='Preeclampsia/Eclampsia')
            Y = data['Preeclampsia/Eclampsia']
        else:
            self.data = data
            X = data.drop(columns='Label')
            Y = data['Label']

        return X, Y

    def normalizeData(self, data, method):

        if method == 'MinMax':
            scaler = MinMaxScaler()
        elif method == 'StandardScale':
            scaler = StandardScaler()

        data_imputed = scaler.fit_transform(data)
        X_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)
        self.data = X_imputed_df

        return X_imputed_df

    def imputeData(self, data1, data2=None):
        MI_Imp = IterativeImputer()  # Scikitlearn's Iterative imputer

        if (data2 is not None):  # If running both datasets
            if (data1.isnull().values.any() == True | data2.isnull().values.any() == True):
                data = data1.append(data2)
                self.data = pd.DataFrame(np.round(MI_Imp.fit_transform(data)), columns=data.columns)
            else:
                self.data = data1.append(data2)
        else:
            if (data1.isnull().values.any() == True):
                self.data = pd.DataFrame(np.round(MI_Imp.fit_transform(data1)), columns=data1.columns)
            else:
                self.data = data1

    def detectOutliers(self, method, con):

        print(self.X_train.shape, self.Y_train.shape)

        if method == 'iso':
            out = IsolationForest(contamination=con)
        elif method == 'lof':
            out = LocalOutlierFactor(contamination=con)
        elif method == 'ocsvm':
            out = OneClassSVM(nu=0.01)
        elif method == 'ee':
            out = EllipticEnvelope(contamination=con)

        yhat = out.fit_predict(self.X_train)

        # select all rows that are not outliers
        mask = (yhat != -1)

        self.X_train = self.X_train.loc[mask]
        self.Y_train = self.Y_train.loc[mask]

        print(self.X_train.shape, self.Y_train.shape)

    def featureSelection(self, numFeatures, method):


        self.method = method

        if method == 1:
            model = XGBClassifier()
            model.fit(self.X_train, self.Y_train)

            # Save graph
            ax = plot_importance(model, max_num_features=numFeatures)
            image = pyplot.gcf()
            neptune.log_image('XGBFeatures', image, image_name='XGBFeatures')

            # Get and save best features
            feature_important = model.get_booster().get_fscore()
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
            XGBoostFeatures = list(data.index[0:numFeatures])
            return XGBoostFeatures

        if method == 2:
            # instantiate SelectKBest to determine 20 best features
            fs = SelectKBest(score_func=chi2, k=numFeatures)
            fs.fit(self.X_train, self.Y_train)
            df_scores = pd.DataFrame(fs.scores_)
            df_columns = pd.DataFrame(self.X_train.columns)
            # concatenate dataframes
            feature_scores = pd.concat([df_columns, df_scores], axis=1)
            feature_scores.columns = ['Feature_Name', 'Chi2 Score']  # name output columns
            feature_scores.sort_values(by=['Chi2 Score'], ascending=False, inplace=True)
            features = feature_scores.iloc[0:numFeatures]
            chi2Features = features['Feature_Name']
            self.Chi2features = features
            return chi2Features

        if method == 3:
            # Mutual Information features
            fs = SelectKBest(score_func=mutual_info_classif, k=numFeatures)
            fs.fit(self.X_train, self.Y_train)
            df_scores = pd.DataFrame(fs.scores_)
            df_columns = pd.DataFrame(self.X_train.columns)
            # concatenate dataframes
            feature_scores = pd.concat([df_columns, df_scores], axis=1)
            feature_scores.columns = ['Feature_Name', 'MI Score']  # name output columns
            feature_scores.sort_values(by=['MI Score'], ascending=False, inplace=True)
            features = feature_scores.iloc[0:numFeatures]
            mutualInfoFeatures = features['Feature_Name']
            self.MIFeatures = features
            return mutualInfoFeatures

        if method == 4:
            features = self.X_train.columns
            return features

    def setData(self, X_train, X_test, Y_train, Y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def buildModel(self, topFeatures):

        LOG_DIR = f"{int(time.time())}"

        # Set all to numpy arrays
        self.X_train = self.X_train[topFeatures].to_numpy()
        self.Y_train = self.Y_train.to_numpy()
        self.X_val = self.X_val[topFeatures].to_numpy()
        self.Y_val = self.Y_val.to_numpy()
        self.X_test = self.X_test[topFeatures].to_numpy()
        self.Y_test = self.Y_test.to_numpy()

        inputSize = self.X_train.shape[1]


        self.training_generator = BalancedBatchGenerator(self.X_train, self.Y_train,
                                                         batch_size=self.PARAMS['batch_size'],
                                                         sampler=RandomOverSampler(),
                                                         random_state=42)

        self.validation_generator = BalancedBatchGenerator(self.X_test, self.Y_test,
                                                           batch_size=self.PARAMS['batch_size'],
                                                           sampler=RandomOverSampler(),
                                                           random_state=42)
        # define the keras model
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(inputSize,)))

        # Hidden Layers
        for i in range(self.PARAMS['num_layers']):
            self.model.add(
                Dense(units=self.PARAMS['units_' + str(i)], activation=self.PARAMS['dense_activation_' + str(i)]))
            if self.PARAMS['Dropout']:
                self.model.add(Dropout(self.PARAMS['Dropout_Rate']))
            if self.PARAMS['BatchNorm']:
                self.model.add(BatchNormalization(momentum=self.PARAMS['Momentum']))

        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.Y_train), self.Y_train)
        class_weight_dict = dict(enumerate(class_weights))
        pos = class_weight_dict[1]
        neg = class_weight_dict[0]
        bias = np.log(pos / neg)

        if self.PARAMS['bias_init'] == 0:
            # Final Layer
            self.model.add(Dense(1, activation=self.PARAMS['final_activation']))

        elif self.PARAMS['bias_init'] == 1:
            # Final Layer
            self.model.add(Dense(1, activation=self.PARAMS['final_activation'],
                                 bias_initializer=tf.keras.initializers.Constant(
                                     value=bias)))

        # Reset class weights for use in loss function
        scalar = len(self.Y_train)
        class_weight_dict[0] = scalar / self.Y_train.value_counts()[0]
        class_weight_dict[1] = scalar / self.Y_train.value_counts()[1]

        # Loss Function
        if self.PARAMS['focal']:
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
        else:
            loss = 'binary_crossentropy'

        # Compilation
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.PARAMS['learning_rate']),
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        # Question - Can you put a list in here?

        self.history = self.model.fit(self.training_generator,
                                      epochs=self.PARAMS['epochs'], validation_data=(self.validation_generator),
                                      verbose=2, class_weight=class_weight_dict, callbacks=[NeptuneMonitor()])

    def evaluateModel(self):

        # Graphing results
        plt.clf()
        plt.cla()
        plt.close()

        auc = plt.figure()
        # plt.ylim(0.40, 0.66)
        plt.plot(self.history.history['auc'])
        plt.title('model auc')
        plt.ylabel('auc')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        neptune.log_image('AUC/Epochs', auc, image_name='aucPlot')

        plt.clf()
        plt.cla()
        plt.close()

        loss = plt.figure()
        # plt.ylim(0.0, 0.15)
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        neptune.log_image('Loss/Epochs', loss, image_name='lossPlot')
        # plt.show()

        # y_predict = self.best_model.predict_classes(self.test_X) # deprecated

        y_predict = (self.model.predict(self.X_test) > 0.5).astype("int32")

        self.specificity = specificity_score(self.Y_test, y_predict)

        gmean = geometric_mean_score(self.Y_test, y_predict)

        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        self.predictedNo = y_predict.sum()
        self.trueNo = self.Y_test.sum()
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.Y_test, y_predict).ravel()

        Results = {"Loss": score[0],
                   "Accuracy": score[1],
                   "AUC": score[4],
                   "Gmean": gmean,
                   "Recall": score[3],
                   "Precision": score[2],
                   "Specificity": self.specificity,
                   "True Positives": self.tp,
                   "True Negatives": self.tn,
                   "False Positives": self.fp,
                   "False Negatives": self.fn,
                   "History": self.history}

        print(f'Total Cases: {len(y_predict)}')
        print(f'Predict #: {y_predict.sum()} / True # {self.Y_test.sum()}')
        print(f'True Positives #: {self.tp} / True Negatives # {self.tn}')
        print(f'False Positives #: {self.fp} / False Negatives # {self.fn}')
        print(f'Test loss: {score[0]:.6f} / Test accuracy: {score[1]:.6f} / Test AUC: {score[4]:.6f}')
        print(f'Test Recall: {score[3]:.6f} / Test Precision: {score[2]:.6f}')
        print(f'Test Specificity: {self.specificity:.6f}')
        print(f'Test Gmean: {gmean:.6f}')

        # Feature Selection
        if self.method == 2:
            log_table('Chi2features', self.Chi2features)

        elif self.method == 3:
            log_table('MIFeatures', self.MIFeatures)

        return Results

class NoGen(fullNN):
    def __init__(self, PARAMS):

        self.PARAMS = PARAMS

    def buildModel(self, topFeatures):

        LOG_DIR = f"{int(time.time())}"

        # Set all to numpy arrays
        self.X_train = self.X_train[topFeatures]
        self.Y_train = self.Y_train
        self.X_test = self.X_test[topFeatures]
        self.Y_test = self.Y_test

        inputSize = self.X_train.shape[1]


        # define the keras model
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(inputSize,)))

        # Hidden Layers
        for i in range(self.PARAMS['num_layers']):
            self.model.add(
                Dense(units=self.PARAMS['units_' + str(i)], activation=self.PARAMS['dense_activation_' + str(i)]))
            if self.PARAMS['Dropout']:
                     self.model.add(Dropout(self.PARAMS['Dropout_Rate']))
            if self.PARAMS['BatchNorm']:
                     self.model.add(BatchNormalization(momentum=self.PARAMS['Momentum']))

        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.Y_train), self.Y_train)
        class_weight_dict = dict(enumerate(class_weights))


        pos = self.Y_train.value_counts()[0]
        neg = self.Y_train.value_counts()[1]
        bias = np.log(pos / neg)

        if self.PARAMS['bias_init'] == 0:
            # Final Layer
            self.model.add(Dense(1, activation=self.PARAMS['final_activation']))

        elif self.PARAMS['bias_init'] == 1:
            # Final Layer
            self.model.add(Dense(1, activation=self.PARAMS['final_activation'],
                                 bias_initializer=tf.keras.initializers.Constant(
                                     value=bias)))

        # Reset class weights for use in loss function
        scalar = len(self.Y_train)
        weight_for_0 = (1 / self.Y_train.value_counts()[0]) * (scalar) / 2.0
        weight_for_1 = (1 / self.Y_train.value_counts()[1]) * (scalar) / 2.0
        class_weight_dict = {0: weight_for_0, 1: weight_for_1}



        # Loss Function
        if self.PARAMS['focal']:
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
        else:
            loss = 'binary_crossentropy'

        # Compilation
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.PARAMS['learning_rate']),
                           loss=weighted_binary_cross_entropy(class_weight_dict),
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.PARAMS['batch_size'],
                                      epochs=self.PARAMS['epochs'],
                                      verbose=2, class_weight=class_weight_dict, callbacks=[NeptuneMonitor()])

if __name__ == "__main__":

    PARAMS = {'num_layers': 3,
              'dense_activation_0': 'tanh',
              'dense_activation_1': 'relu',
              'dense_activation_2': 'relu',
              'units_0': 30,
              'units_1': 30,
              'units_2': 45,
              'final_activation': 'sigmoid',
              'optimizer': 'Adam',
              'learning_rate': 0.001,
              'batch_size': 64,
              'bias_init': 0,
              'epochs': 50,
              'focal': False,
              'alpha': 0.5,
              'gamma': 1.25,
              'class_weights': True,
              'initializer': 'RandomUniform',
              'Dropout': True,
              'Dropout_Rate': 0.40,
              'BatchNorm': True,
              'Momentum': 0.60,
              'Generator': False,
              'MAX_TRIALS': 5}

    neptune.create_experiment(name='Spect', params=PARAMS, send_hardware_metrics=True,
                              tags=['CV'],
                              description='Cross-val Spect')

    #neptune.log_text('my_text_data', 'text I keep track of, like query or tokenized word')

    if PARAMS['Generator'] == False:
        model = NoGen(PARAMS)

    """
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Processed/Oklahoma/Complete/Full/Outliers/Chi2_Categorical.csv')
    """
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)

    data = cleanSpect()
    X, y = model.prepData(age='Categorical', data=data)

    #model.normalizeData(data=X, method='StandardScale')

    aucList = []
    gmeanList = []
    accList = []
    precisionList = []
    recallList = []
    specList = []
    lossList = []
    historyList = []

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.setData(X_train, X_test, y_train, y_test)
        features = model.featureSelection(numFeatures=20, method=4)
        # For hand-tuning
        model.buildModel(features)

        Results = model.evaluateModel()

        aucList.append(Results["AUC"])
        gmeanList.append(Results["Gmean"])
        accList.append(Results["Accuracy"])
        precisionList.append(Results["Precision"])
        recallList.append(Results["Recall"])
        specList.append(Results["Specificity"])
        lossList.append(Results["Loss"])
        historyList.append(Results["History"])  # List of lists, will each entry history of a particular run


    # Get Average Results
    lossMean = statistics.mean(lossList)
    aucMean = statistics.mean(aucList)
    gmeanMean = statistics.mean(gmeanList)
    accMean = statistics.mean(accList)
    specMean = statistics.mean(specList)
    recallMean = statistics.mean(recallList)
    precMean = statistics.mean(precisionList)

    neptune.log_metric('Mean loss', lossMean)
    neptune.log_metric('Mean accuracy', accMean)
    neptune.log_metric('Mean AUC', aucMean)
    neptune.log_metric('Mean specificity', specMean)
    neptune.log_metric('Mean recall', recallMean)
    neptune.log_metric('Mean precision', precMean)
    neptune.log_metric('Mean gmean', gmeanMean)


    def plotAvg(historyList):
        aucAvg = []
        lossAvg = []

        for i in range(len(historyList[0].history['auc'])): # Iterate through each epoch
            # Clear list
            auc = []
            aucVal = []
            loss = []
            lossVal = []

            for j in range(len(historyList)): # Iterate through each history object

                # Append each model's measurement for epoch i
                auc.append(historyList[j].history['auc'][i])
                loss.append(historyList[j].history['loss'][i])

            # Once get measurement for each model, get average measurement for that epoch
            aucAvg.append(statistics.mean(auc))
            lossAvg.append(statistics.mean(loss))

        # Graphing results
        plt.clf()
        plt.cla()
        plt.close()

        avgauc = plt.figure()
        # plt.ylim(0.40, 0.66)
        plt.plot(aucAvg)
        plt.title('model auc')
        plt.ylabel('auc')
        plt.xlabel('epoch')
        plt.legend(['training'], loc='upper right')
        neptune.log_image('Average AUC', avgauc, image_name='avgAucPlot')

        plt.clf()
        plt.cla()
        plt.close()

        avgloss = plt.figure()
        # plt.ylim(0.40, 0.66)
        plt.plot(lossAvg)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training'], loc='upper right')
        neptune.log_image('Average Loss', avgloss, image_name='avgLossPlot')

    plotAvg(historyList)
