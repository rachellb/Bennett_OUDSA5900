#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For handling data
import pandas as pd
import numpy as np
from datetime import datetime
import os
from Cleaning.Clean import *
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.utils import class_weight
from statistics import mean

# For imputing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow.keras.backend as K
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
from secret import api_
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from neptune.new.types import File

# For Reproducibility
import random

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

    def __init__(self, PARAMS, name=None):

        self.PARAMS = PARAMS
        self.name = name

    def normalizeData(self, data):

        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data

    def imputeData(self):
        # Scikitlearn's Iterative imputer
        # Default imputing method is Bayesian Ridge Regression

        if self.PARAMS['estimator'] == "BayesianRidge":
            estimator = BayesianRidge()
        elif self.PARAMS['estimator'] == "DecisionTree":
            estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0)
        elif self.PARAMS['estimator'] == "ExtraTrees":
            estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
        elif self.PARAMS['estimator'] == "KNN":
            estimator = KNeighborsRegressor(n_neighbors=15)

        MI_Imp = IterativeImputer(random_state=0, estimator=estimator)

        if (self.name == 'MOMI'):
            if self.data.isnull().values.any():
                self.X_train_imputed = pd.DataFrame(MI_Imp.fit_transform(self.X_train), columns=self.X_train.columns)
                self.X_test_imputed = pd.DataFrame(MI_Imp.transform(self.X_test), columns=self.X_test.columns)

                # Rounding only the categorical variables that were imputed
                self.X_train = self.X_train_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
                                                           'DeliveriesPriorAdmission': 0, 'TotalAbortions': 0,
                                                           'Primagrivada': 0, 'MaternalNeuromuscularDisease': 0,
                                                           'KidneyDisease': 0, 'Thrombocytopenia': 0, 'InfSex': 0,
                                                           'CNSAbnormality': 0, 'CongenitalSyphilis': 0, 'UTI': 0,
                                                           'RaceCollapsed': 0, 'Systolic': 0})

                self.X_test = self.X_test_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
                                                         'DeliveriesPriorAdmission': 0, 'TotalAbortions': 0,
                                                         'Primagrivada': 0, 'MaternalNeuromuscularDisease': 0,
                                                         'KidneyDisease': 0, 'Thrombocytopenia': 0, 'InfSex': 0,
                                                         'CNSAbnormality': 0, 'CongenitalSyphilis': 0, 'UTI': 0,
                                                         'RaceCollapsed': 0, 'Systolic': 0})

            # Fix incorrectly imputed value
            self.X_train['RaceCollapsed'] = np.where(((self.X_train['RaceCollapsed'] > 4)), 4,
                                                     self.X_train['RaceCollapsed'])
            self.X_test['RaceCollapsed'] = np.where(((self.X_test['RaceCollapsed'] > 4)), 4,
                                                    self.X_test['RaceCollapsed'])

            # Fix incorrectly imputed value
            self.X_train['RaceCollapsed'] = np.where(((self.X_train['RaceCollapsed'] < 0)), 0,
                                                     self.X_train['RaceCollapsed'])
            self.X_test['RaceCollapsed'] = np.where(((self.X_test['RaceCollapsed'] < 0)), 0,
                                                    self.X_test['RaceCollapsed'])

        else:
            self.X_train = pd.DataFrame(np.round(MI_Imp.fit_transform(self.X_train)), columns=self.X_train.columns)
            self.X_test = pd.DataFrame(np.round(MI_Imp.transform(self.X_test)), columns=self.X_test.columns)

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

    def featureSelection(self):

        # If there are less features than the number selected
        numFeatures = min(self.PARAMS['Feature_Num'], (self.X_train.shape[1]))

        if self.PARAMS['Feature_Selection'] == "XGBoost":
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
            topFeatures = list(data.index[0:numFeatures])

        if self.PARAMS['Feature_Selection'] == "Chi2":

            self.X_train[self.X_train < 0] = 0
            self.X_test[self.X_test < 0] = 0


            # instantiate SelectKBest to determine 20 best features
            fs = SelectKBest(score_func=chi2, k=numFeatures)
            fs.fit(self.X_train, self.Y_train)
            df_scores = pd.DataFrame({"Scores": fs.scores_, "P-values": fs.pvalues_})
            df_columns = pd.DataFrame(self.X_train.columns)
            # concatenate dataframes
            feature_scores = pd.concat([df_columns, df_scores], axis=1)
            feature_scores.columns = ['Feature_Name', 'Chi2 Score', 'P-value']  # name output columns
            feature_scores.sort_values(by=['Chi2 Score'], ascending=False, inplace=True)
            features = feature_scores.iloc[0:numFeatures]
            chi2Features = features['Feature_Name']
            topFeatures = list(chi2Features)
            self.Chi2Features = features

        if self.PARAMS['Feature_Selection'] == "MI":
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
            topFeatures = list(mutualInfoFeatures)
            self.MIFeatures = mutualInfoFeatures

        self.X_train = self.X_train[topFeatures]
        self.X_test = self.X_test[topFeatures]

    def prepData(self, data):

        # Trying to save memory by changing binary variables to int8
        for col in list(data.columns):
            if list(data[col].unique()) == [0, 1]:
                data[col] = data[col].astype('int8')

        X = data.drop(columns='Preeclampsia/Eclampsia')
        Y = data['Preeclampsia/Eclampsia']

        return X, Y

    def setData(self, X_train, X_test, Y_train, Y_test, subgroup=None):

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        # Double check
        if subgroup == 'Native':
            idx = self.X_test.index[self.X_test['Native American'] == 1]
            self.X_test = self.X_test.loc[idx]
            self.Y_test = self.Y_test.loc[idx]

        elif subgroup == 'African':
            idx = self.X_test.index[self.X_test['Black'] == 1]
            self.X_test = self.X_test.loc[idx]
            self.Y_test = self.Y_test.loc[idx]

    def buildModel(self):

        LOG_DIR = f"{int(time.time())}"

        # Set all to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy()
        self.X_test = self.X_test.to_numpy()
        self.Y_test = self.Y_test.to_numpy()

        inputSize = self.X_train.shape[1]


        self.training_generator = BalancedBatchGenerator(self.X_train, self.Y_train,
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

        # Loss Function
        if self.PARAMS['focal']:
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
        elif self.PARAMS['class_weights']:
            loss = weighted_binary_cross_entropy(class_weight_dict)
        else:
            loss = 'binary_crossentropy'

        # Conditional for each optimizer
        if self.PARAMS['optimizer'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'NAdam':
            optimizer = tf.keras.optimizers.Nadam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        # Compilation
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        self.history = self.model.fit(self.training_generator, epochs=self.PARAMS['epochs'], verbose=2)

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
        run['AUC/Epochs'].upload(auc)

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
        run['Loss/Epochs'].upload(loss)
        # plt.show()

        # y_predict = self.best_model.predict_classes(self.test_X) # deprecated

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
                   #"History": self.history}

        print(f'Total Cases: {len(y_predict)}')
        print(f'Predict #: {y_predict.sum()} / True # {self.Y_test.sum()}')
        print(f'True Positives #: {tp} / True Negatives # {tn}')
        print(f'False Positives #: {fp} / False Negatives # {fn}')
        print(f'Test loss: {score[0]:.6f} / Test accuracy: {score[1]:.6f} / Test AUC: {score[4]:.6f}')
        print(f'Test Recall: {score[3]:.6f} / Test Precision: {score[2]:.6f}')
        print(f'Test Specificity: {specificity:.6f}')
        print(f'Test Gmean: {gmean:.6f}')

        # Feature Selection
        if self.PARAMS['Feature_Selection'] == "XGBoost":
            # TODO: figure out how to load and save this image
            image = Image.open(self.dataset + 'XGBoostTopFeatures.png')
            neptune.log_image('XGBFeatures', image, image_name='XGBFeatures')

        elif self.PARAMS['Feature_Selection'] == "Chi2":
            run['Chi2features'].upload(File.as_html(self.Chi2Features))


        elif self.PARAMS['Feature_Selection'] == "MI":
            run['MIFeatures'].upload(File.as_html(self.MIFeatures))

        return Results



class NoGen(fullNN):
    def __init__(self, PARAMS, name=None):

        self.PARAMS = PARAMS
        self.name = name

    def buildModel(self):

        LOG_DIR = f"{int(time.time())}"

        # Set all to numpy arrays
        self.X_train = self.X_train
        self.Y_train = self.Y_train
        self.X_test = self.X_test
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

        # Loss Function
        if self.PARAMS['focal']:
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
        elif self.PARAMS['class_weights']:
            loss = weighted_binary_cross_entropy(class_weight_dict)
        else:
            loss = 'binary_crossentropy'

        # Conditional for each optimizer
        if self.PARAMS['optimizer'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'NAdam':
            optimizer = tf.keras.optimizers.Nadam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        # Compilation
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.PARAMS['batch_size'],
                                      epochs=self.PARAMS['epochs'],
                                      verbose=2, callbacks=[neptune_cbk])

if __name__ == "__main__":

    # Set seeds
    def reset_random_seeds():
        os.environ['PYTHONHASHSEED'] = str(1)
        tf.random.set_seed(1)
        np.random.seed(1)
        random.seed(1)

    #reset_random_seeds()

    start_time = time.time()

    PARAMS = {'num_layers': 3,
              'dense_activation_0': 'tanh',
              'dense_activation_1': 'relu',
              'dense_activation_2': 'relu',
              'units_0': 60,
              'units_1': 30,
              'units_2': 45,
              'final_activation': 'sigmoid',
              'optimizer': 'RMSprop',
              'learning_rate': 0.001,
              'batch_size': 8192,
              'bias_init': 0,
              'epochs': 30,
              'focal': True,
              'alpha': 0.95,
              'gamma': 1,
              'class_weights': False,
              'initializer': 'RandomUniform',
              'Dropout': True,
              'Dropout_Rate': 0.20,
              'BatchNorm': False,
              'Momentum': 0.60,
              'Feature_Selection': 'Chi2',
              'Feature_Num': 20,
              'estimator': "BayesianRidge",
              'Generator': False,
              'MAX_TRIALS': 5}

    run = neptune.init(project='rachellb/CVPreeclampsia',
                       api_token=api_,
                       name='Oklahoma Full',
                       tags=['Focal Loss', 'Hyperband', 'Updated', 'Testing subgroups', '350CV'],
                       source_files=['NeuralNetwork_NoTune.py', 'Cleaning/Clean.py'])

    run['hyper-parameters'] = PARAMS
    #neptune.log_text('my_text_data', 'text I keep track of, like query or tokenized word')

    if PARAMS['Generator'] == False:
        model = NoGen(PARAMS)

    else:
        model = fullNN(PARAMS)

    # Get data
    #parent = os.path.dirname(os.getcwd())
    #dataPath = os.path.join(parent, 'Data/Processed/Texas/Full/Outliers/Complete/Chi2_Categorical_041521.csv')
    #dataPath = os.path.join(parent, 'Data/Processed/Oklahoma/Complete/Full/Outliers/Chi2_Categorical_042021.csv')
    #dataPath = os.path.join(parent, 'Data/Processed/Texas/Native/Chi2_Categorical_041521.csv')


    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=35, random_state=36851234)

    ok2017, ok2018, african, native = cleanDataOK(dropMetro=True)
    full = ok2017.append(ok2018)
    X, y = model.prepData(full)

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
        model.setData(X_train, X_test, y_train, y_test, subgroup='African')


        model.imputeData()
        model.featureSelection()
        # For hand-tuning
        model.buildModel()
        Results = model.evaluateModel()

        aucList.append(Results["AUC"])
        gmeanList.append(Results["Gmean"])
        accList.append(Results["Accuracy"])
        precisionList.append(Results["Precision"])
        recallList.append(Results["Recall"])
        specList.append(Results["Specificity"])
        lossList.append(Results["Loss"])
        #historyList.append(Results["History"])  # List of lists, will each entry history of a particular run
        tpList.append(Results['True Positives'])
        tnList.append(Results['True Negatives'])
        fpList.append(Results['False Positives'])
        fnList.append(Results['False Negatives'])



    run['List loss'] = lossList
    run['List accuracy'] = accList
    run['List AUC'] = aucList
    run['List specificity'] = specList
    run['List recall'] = recallList
    run['List precisio'] = precisionList
    run['List gmean'] = gmeanList
    run['List TP'] = tpList
    run['List TN'] = tnList
    run['List FP'] = fpList
    run['List FN'] = fnList


    # Get Average Results
    lossMean = statistics.mean(lossList)
    aucMean = statistics.mean(aucList)
    gmeanMean = statistics.mean(gmeanList)
    accMean = statistics.mean(accList)
    specMean = statistics.mean(specList)
    recallMean = statistics.mean(recallList)
    precMean = statistics.mean(precisionList)
    tpMean = statistics.mean(tpList)
    tnMean = statistics.mean(tnList)
    fpMean = statistics.mean(fpList)
    fnMean = statistics.mean(fnList)


    run['Mean loss'] = lossMean
    run['Mean accuracy'] = accMean
    run['Mean AUC'] = aucMean
    run['Mean specificity'] = specMean
    run['Mean recall'] = recallMean
    run['Mean precisio'] = precMean
    run['Mean gmean'] = gmeanMean
    run['Mean TP'] = tpMean
    run['Mean TN'] = tnMean
    run['Mean FP'] = fpMean
    run['Mean FN'] = fnMean

    # Get Standard Deviation of Results
    lossSD = statistics.pstdev(lossList)
    aucSD = statistics.pstdev(aucList)
    gmeanSD = statistics.pstdev(gmeanList)
    accSD = statistics.pstdev(accList)
    specSD = statistics.pstdev(specList)
    recallSD = statistics.pstdev(recallList)
    precSD = statistics.pstdev(precisionList)
    tpSD = statistics.pstdev(tpList)
    fpSD = statistics.pstdev(fpList)
    tnSD = statistics.pstdev(tnList)
    fnSD = statistics.pstdev(fnList)


    run['SD loss'] = lossSD
    run['SD accuracy'] = accSD
    run['SD AUC'] = aucSD
    run['SD specificity'] = specSD
    run['SD recall'] = recallSD
    run['SD precision'] = precSD
    run['SD gmean'] = gmeanSD
    run['SD tp'] = tpSD
    run['SD tn'] = fpSD
    run['SD fp'] = tnSD
    run['SD fn'] = fnSD


    def plotAvg(historyList):
        aucAvg = []
        lossAvg = []

        for i in range(len(historyList[0].history['auc'])): # Iterate through each epoch
            # Clear list
            auc = []
            loss = []

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
        run['Average AUC'].upload(avgauc)

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
        run['Average Loss'].upload(avgloss)

    #plotAvg(historyList)

    mins = (time.time() - start_time) / 60  # Time in seconds

    run['minutes'] = mins

    run.stop()