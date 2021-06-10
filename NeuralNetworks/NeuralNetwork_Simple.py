#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this particular network does no tuning or cross validation.


# For handling data
from datetime import datetime
import os

from sklearn.utils import class_weight

# For imputing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

# For Encoding and Preprocessing
from sklearn import preprocessing

# For feature selection
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_selection import f_classif

# For balancing batches
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler

# For NN and tuning
import tensorflow as tf
import tensorflow.keras.backend as K
import kerastuner
from kerastuner.tuners import Hyperband, BayesianOptimization, RandomSearch
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


# For additional metrics
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import confusion_matrix
import tensorflow_addons as tfa  # For focal loss function
import time
import matplotlib.pyplot as plt
from PIL import Image


#For Outlier Detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

date = datetime.today().strftime('%m%d%y')  # For labelling purposes

# For recording results
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from neptune.new.types import File
from secret import api_
from Cleaning.Clean import *


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

    def prepData(self, data, age=None):

        if (self.name=="MOMI"):
            data = pd.read_csv(data)
            self.data = data
        else:
            self.age = age
            data = pd.read_csv(data)

        return data

    def normalizeData(self):

        if self.PARAMS['Normalize'] == 'MinMax':
            scaler = MinMaxScaler()
        elif self.PARAMS['Normalize'] == 'StandardScale':
            scaler = StandardScaler()


        if (self.name=="MOMI"):
            # Fit and transform training data, then transform val and test using info gained from fitting
            scaleColumns = ['MotherAge', 'WeightAtAdmission',
                            'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions', 'WeightAtAdmission',
                            'PNV_GestAge', 'PNV_Weight_Oz', 'Systolic', 'Prev_highBP']

            self.X_train[scaleColumns] = scaler.fit_transform(self.X_train[scaleColumns])
            self.X_val[scaleColumns] = scaler.fit_transform(self.X_val[scaleColumns])
            self.X_test[scaleColumns] = scaler.fit_transform(self.X_test[scaleColumns])

        else:
            data_imputed = scaler.fit_transform(data)
            X_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)
            self.data = X_imputed_df


    def encodeData(self):

        encodeCols = ['Insurance', 'MaternalNeuromuscularDisease', 'MCollagenVascularDisease',
                      'MStructuralHeartDiseas', 'MPostPartumComplications', 'DiabetesMellitus', 'ThyroidDisease',
                      'MLiverGallPanc', 'KidneyDisease', 'MAnemiaWOHemoglobinopathy', 'MHemoglobinopathy',
                      'Thrombocytopenia', 'ViralOrProtoInf', 'OtherSubstanceAbuse', 'InfSex', 'CNSAbnormality',
                      'RaceCollapsed']

        ohe = OneHotEncoder()

        # Train on the categorical variables
        ohe.fit(self.data[encodeCols])

        X_trainCodes = ohe.transform(self.X_train[encodeCols]).toarray()
        X_valCodes = ohe.transform(self.X_val[encodeCols]).toarray()
        X_testCodes = ohe.transform(self.X_test[encodeCols]).toarray()
        feature_names = ohe.get_feature_names(encodeCols)

        self.X_train = pd.concat([self.X_train.drop(columns=encodeCols),
                                  pd.DataFrame(X_trainCodes, columns=feature_names).astype(int)], axis=1)

        self.X_val = pd.concat([self.X_val.drop(columns=encodeCols),
                                  pd.DataFrame(X_valCodes, columns=feature_names).astype(int)], axis=1)

        self.X_test = pd.concat([self.X_test.drop(columns=encodeCols),
                                  pd.DataFrame(X_testCodes, columns=feature_names).astype(int)], axis=1)



    def imputeData(self, data1=None, data2=None):
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


        if (self.name=='MOMI'):
            if self.data.isnull().values.any():
                self.X_train_imputed = pd.DataFrame(MI_Imp.fit_transform(self.X_train), columns=self.X_train.columns)
                self.X_val_imputed = pd.DataFrame(MI_Imp.transform(self.X_val), columns=self.X_val.columns)
                self.X_test_imputed = pd.DataFrame(MI_Imp.transform(self.X_test), columns=self.X_test.columns)

                # Rounding only the categorical variables that were imputed
                self.X_train = self.X_train_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
                                                           'DeliveriesPriorAdmission': 0, 'TotalAbortions': 0,
                                                           'Primagrivada': 0, 'MaternalNeuromuscularDisease': 0,
                                                           'KidneyDisease': 0, 'Thrombocytopenia': 0, 'InfSex': 0,
                                                           'CNSAbnormality': 0, 'CongenitalSyphilis': 0, 'UTI': 0,
                                                           'RaceCollapsed': 0, 'Systolic': 0})
                self.X_val = self.X_val_imputed.round({'Insurance': 0, 'TotalNumPregnancies': 0,
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
            self.X_val['RaceCollapsed'] = np.where(((self.X_val['RaceCollapsed'] > 4)), 4,
                                                   self.X_val['RaceCollapsed'])
            self.X_test['RaceCollapsed'] = np.where(((self.X_test['RaceCollapsed'] > 4)), 4,
                                                    self.X_test['RaceCollapsed'])

            # Fix incorrectly imputed value
            self.X_train['RaceCollapsed'] = np.where(((self.X_train['RaceCollapsed'] < 0)), 0,
                                                     self.X_train['RaceCollapsed'])
            self.X_val['RaceCollapsed'] = np.where(((self.X_val['RaceCollapsed'] < 0)), 0,
                                                   self.X_val['RaceCollapsed'])
            self.X_test['RaceCollapsed'] = np.where(((self.X_test['RaceCollapsed'] < 0)), 0,
                                                    self.X_test['RaceCollapsed'])

        else:
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

    def splitData(self):

        self.split1=5
        self.split2=107
        X = self.data.drop(columns='Preeclampsia/Eclampsia')
        Y = self.data['Preeclampsia/Eclampsia']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, stratify=Y,
                                                                                test_size=self.PARAMS['TestSplit'],
                                                                                random_state=self.split1)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              stratify=self.Y_train,
                                                                              test_size=self.PARAMS['ValSplit'],
                                                                              random_state=self.split2)

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

    def featureSelection(self, numFeatures, encode=False):

        # If there are less features than the number selected
        numFeatures = min(numFeatures, (self.X_train.shape[1]))


        if self.PARAMS['Feature_Selection'] == "XGBoost":
            model = XGBClassifier()
            model.fit(self.X_train, self.Y_train)

            # Save graph
            ax = plot_importance(model, max_num_features=numFeatures)
            fig1 = pyplot.gcf()
            #pyplot.show()

            fig1.savefig('XGBoostTopFeatures.png', bbox_inches='tight')

            # Get and save best features
            feature_important = model.get_booster().get_fscore()
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
            XGBoostFeatures = list(data.index[0:numFeatures])

            return XGBoostFeatures

        if self.PARAMS['Feature_Selection'] == "Chi2":
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


        if self.PARAMS['Feature_Selection'] == None:
            topFeatures = self.X_train.columns


        self.X_train = self.X_train[topFeatures]
        self.X_val = self.X_val[topFeatures]
        self.X_test = self.X_test[topFeatures]

    def buildModel(self, topFeatures):


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

        self.validation_generator = BalancedBatchGenerator(self.X_val, self.Y_val,
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

        # Conditional for each optimizer
        if self.PARAMS['optimizer'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'NAdam':
            optimizer = tf.keras.optimizers.Nadam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        # Loss Function
        if self.PARAMS['focal']:
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
        elif self.PARAMS['class_weights']:
            loss = weighted_binary_cross_entropy(class_weight_dict)
        else:
            loss = 'binary_crossentropy'


        # Compilation
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        from neptunecontrib.monitoring.keras import NeptuneMonitor

        self.history = self.model.fit(self.training_generator, epochs=self.PARAMS['epochs'],
                                      validation_data=(self.validation_generator),
                                      verbose=2,  callbacks=[NeptuneMonitor()])

    def evaluateModel(self):

        # Graphing results
        plt.clf()
        plt.cla()
        plt.close()

        auc = plt.figure()
        # plt.ylim(0.40, 0.66)
        plt.plot(self.history.history['auc'])
        plt.plot(self.history.history['val_auc'])
        plt.title('model auc')
        plt.ylabel('auc')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        run['AUC/Epochs'].upload(auc)

        plt.clf()
        plt.cla()
        plt.close()

        loss = plt.figure()
        # plt.ylim(0.0, 0.15)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        run['Loss/Epochs'].upload(loss)
        # plt.show()

        # y_predict = self.best_model.predict_classes(self.test_X) # deprecated

        y_predict = (self.model.predict(self.X_test) > 0.5).astype("int32")

        self.specificity = specificity_score(self.Y_test, y_predict)

        self.gmean = geometric_mean_score(self.Y_test, y_predict)

        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        self.loss = score[0]
        self.accuracy = score[1]
        self.AUC = score[4]
        self.predictedNo = y_predict.sum()
        self.trueNo = self.Y_test.sum()
        self.recall = score[3]
        self.precision = score[2]
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.Y_test, y_predict).ravel()

        run['loss'] = self.loss
        run['accuracy'] = self.accuracy
        run['Test AUC'] = self.AUC
        run['Test AUC(PR)'] = score[5]
        run['specificity'] = self.specificity
        run['recall'] = self.recall
        run['precision'] = self.precision
        run['gmean'] = self.gmean
        run['True Positive'] = self.tp
        run['True Negative'] = self.tn
        run['False Positive'] = self.fp
        run['False Negative'] = self.fn



        print(f'Total Cases: {len(y_predict)}')
        print(f'Predict #: {y_predict.sum()} / True # {self.Y_test.sum()}')
        print(f'True Positives #: {self.tp} / True Negatives # {self.tn}')
        print(f'False Positives #: {self.fp} / False Negatives # {self.fn}')
        print(f'Test loss: {score[0]:.6f} / Test accuracy: {score[1]:.6f} / Test AUC: {score[4]:.6f}')
        print(f'Test Recall: {score[3]:.6f} / Test Precision: {score[2]:.6f}')
        print(f'Test Specificity: {self.specificity:.6f}')
        print(f'Test Gmean: {self.gmean:.6f}')

        # Feature Selection
        if self.PARAMS['Feature_Selection'] == 'XGBoost':
            image = Image.open('XGBoostTopFeatures.png')
            neptune.log_image('XGBFeatures', image, image_name='XGBFeatures')


        elif self.PARAMS['Feature_Selection'] == 'Chi2':

            run['Chi2features'].upload(File.as_html(self.Chi2features))

        elif self.PARAMS['Feature_Selection'] == 'MI':

            run['MIFeatures'].upload(File.as_html(self.MIFeatures))


class NoGen(fullNN):
    def __init__(self, PARAMS, name=None):

        self.PARAMS = PARAMS
        self.name = name

    def buildModel(self, topFeatures):

        self.start_time = time.time()

        LOG_DIR = f"{int(time.time())}"

        # Set all to numpy arrays
        self.X_train = self.X_train[topFeatures]
        self.Y_train = self.Y_train
        self.X_val = self.X_val[topFeatures]
        self.Y_val = self.Y_val
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
        # class_weight_dict[0] = scalar / self.Y_train.value_counts()[0]
        # class_weight_dict[1] = scalar / self.Y_train.value_counts()[1]

        weight_for_0 = (1 / self.Y_train.value_counts()[0]) * (scalar) / 2.0
        weight_for_1 = (1 / self.Y_train.value_counts()[1]) * (scalar) / 2.0

        class_weight_dict = {0: weight_for_0, 1: weight_for_1}

        # Conditional for each optimizer
        if self.PARAMS['optimizer'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(self.PARAMS['learning_rate'], clipnorm=0.0001)

        elif self.PARAMS['optimizer'] == 'NAdam':
            optimizer = tf.keras.optimizers.Nadam(self.PARAMS['learning_rate'], clipnorm=0.0001)

        # Loss Function
        if self.PARAMS['focal']:
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])

        elif self.PARAMS['class_weights']:
            loss = weighted_binary_cross_entropy(class_weight_dict)
        else:
            loss = 'binary_crossentropy'

        # Compilation
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC(),
                                    tf.keras.metrics.AUC(curve='PR')])

        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.PARAMS['batch_size'],
                                      epochs=self.PARAMS['epochs'], validation_data=(self.X_val, self.Y_val),
                                      verbose=2, callbacks=[neptune_cbk])



if __name__ == "__main__":

    """
    alpha = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
    gamma = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2]

    for i in alpha:
        for j in gamma:
    """

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
              'bias_init': False,
              'estimator': "BayesianRidge",
              'epochs': 30,
              'focal': True,
              'alpha': 0.89,
              'gamma': 0.25,
              'class_weights': False,
              'initializer': 'RandomUniform',
              'Dropout': True,
              'Dropout_Rate': 0.20,
              'BatchNorm': False,
              'Momentum': 0.60,
              'Normalize': 'MinMax',
              'Feature_Selection': 'Chi2',
              'Generator': False,
              'TestSplit': 0.10,
              'ValSplit': 0.10}

    run = neptune.init(project='rachellb/FocalPre',
                       api_token=api_,
                       name='MOMI Full',
                       tags=['Focal Loss', 'Hand Tuned', 'Test'],
                       source_files=[])

    run['hyper-parameters'] = PARAMS


    if PARAMS['Generator'] == False:
        model = NoGen(PARAMS, name="MOMI")
    else:
        model = fullNN(PARAMS, name="MOMI")


    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Preprocess/momiEncoded_Full_060821.csv')
    data = model.prepData(data=dataPath)
    model.splitData()
    data = model.imputeData()
    model.normalizeData()
    features = model.featureSelection(numFeatures=20, encode=True)
    model.encodeData()
    model.buildModel()
    model.evaluateModel()
    run.stop()

