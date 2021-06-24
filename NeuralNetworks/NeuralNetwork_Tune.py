#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For handling data
from datetime import datetime
import os
import random
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, RobustScaler
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

import neptune
from neptunecontrib.api.table import log_table
from neptunecontrib.monitoring.keras import NeptuneMonitor
import neptunecontrib.monitoring.kerastuner as npt_utils
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

    def scaleData(self):

        if self.PARAMS['Normalize'] == 'MinMax':
            scaler = MinMaxScaler()
        elif self.PARAMS['Normalize'] == 'StandardScale':
            scaler = StandardScaler()
        elif self.PARAMS['Normalize'] == 'Robust':
            scaler = RobustScaler()

        if (self.name=="MOMI"):
            # Fit and transform training data, then transform val and test using info gained from fitting
            scaleColumns = ['MotherAge', 'WeightAtAdmission',
                            'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions', 'WeightAtAdmission',
                            'PNV_GestAge', 'PNV_Weight_Oz', 'MAP', 'Prev_highBP']

            self.X_train[scaleColumns] = scaler.fit_transform(self.X_train[scaleColumns])
            self.X_val[scaleColumns] = scaler.transform(self.X_val[scaleColumns])
            self.X_test[scaleColumns] = scaler.transform(self.X_test[scaleColumns])

        else:
            data_imputed = scaler.fit_transform(data)
            X_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)
            self.data = X_imputed_df

    def normalizeData(self, method='MinMax'):

        if method == 'MinMax':
            scaler = MinMaxScaler()
        elif method == 'StandardScale':
            scaler = StandardScaler()


        if (self.name=="MOMI"):
            # Fit and transform training data, then transform val and test using info gained from fitting
            scaleColumns = ['MotherAge', 'WeightAtAdmission',
                            'TotalNumPregnancies', 'DeliveriesPriorAdmission', 'TotalAbortions', 'WeightAtAdmission',
                            'PNV_GestAge', 'PNV_Weight_Oz', 'MAP', 'Prev_highBP']

            self.X_train[scaleColumns] = scaler.fit_transform(self.X_train[scaleColumns])
            self.X_val[scaleColumns] = scaler.transform(self.X_val[scaleColumns])
            self.X_test[scaleColumns] = scaler.transform(self.X_test[scaleColumns])

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

        selectCat = [c for c in self.X_train.columns if (c in encodeCols)]

        ohe = OneHotEncoder(handle_unknown='ignore')

        # Train on the categorical variables
        ohe.fit(self.data[selectCat])

        X_trainCodes = ohe.transform(self.X_train[selectCat]).toarray()
        X_valCodes = ohe.transform(self.X_val[selectCat]).toarray()
        X_testCodes = ohe.transform(self.X_test[selectCat]).toarray()
        feature_names = ohe.get_feature_names(selectCat)

        self.X_train = pd.concat([self.X_train.drop(columns=selectCat).reset_index(drop=True),
                                  pd.DataFrame(X_trainCodes, columns=feature_names).astype(int).reset_index(drop=True)],
                                 axis=1)

        self.X_val = pd.concat([self.X_val.drop(columns=selectCat).reset_index(drop=True),
                                pd.DataFrame(X_valCodes, columns=feature_names).astype(int).reset_index(drop=True)],
                               axis=1)

        self.X_test = pd.concat([self.X_test.drop(columns=selectCat).reset_index(drop=True),
                                 pd.DataFrame(X_testCodes, columns=feature_names).astype(int).reset_index(drop=True)],
                                axis=1)

        # OHE adds unnecessary nan column, which needs to be dropped
        self.X_train = self.X_train.loc[:, ~self.X_train.columns.str.endswith('_nan')]
        self.X_val = self.X_val.loc[:, ~self.X_val.columns.str.endswith('_nan')]
        self.X_test = self.X_test.loc[:, ~self.X_test.columns.str.endswith('_nan')]

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
        X = self.data.drop(columns='Mild_PE')
        Y = self.data['Mild_PE']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, stratify=Y,
                                                                                test_size=self.PARAMS['TestSplit'],
                                                                                random_state=self.split1)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              stratify=self.Y_train,
                                                                              test_size=self.PARAMS['ValSplit'],
                                                                              random_state=self.split2)

    def detectOutliers(self, con='auto'):

        print(self.X_train.shape, self.Y_train.shape)

        if self.PARAMS['OutlierRemove'] == 'iso':
            out = IsolationForest(contamination=con)
        elif self.PARAMS['OutlierRemove'] == 'lof':
            out = LocalOutlierFactor(contamination=con)
        elif self.PARAMS['OutlierRemove'] == 'ocsvm':
            out = OneClassSVM(nu=0.01)
        elif self.PARAMS['OutlierRemove'] == 'ee':
            out = EllipticEnvelope()

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
            fig1 = pyplot.gcf()
            # pyplot.show()

            fig1.savefig('XGBoostTopFeatures.png', bbox_inches='tight')

            # Get and save best features
            feature_important = model.get_booster().get_fscore()
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
            topFeatures = list(data.index[0:numFeatures])

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
            self.MIFeatures = features

        if self.PARAMS['Feature_Selection'] == None:
            topFeatures = self.X_train.columns

        self.X_train = self.X_train[topFeatures]
        self.X_val = self.X_val[topFeatures]
        self.X_test = self.X_test[topFeatures]

    def hpTuning(self):
        self.start_time = time.time()

        tf.keras.backend.clear_session()

        # Set all to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy()
        self.X_val = self.X_val.to_numpy()
        self.Y_val = self.Y_val.to_numpy()
        self.X_test = self.X_test.to_numpy()
        self.Y_test = self.Y_test.to_numpy()

        inputSize = self.X_train.shape[1]



        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.Y_train), self.Y_train)
        class_weight_dict = dict(enumerate(class_weights))
        pos = class_weight_dict[1]
        neg = class_weight_dict[0]

        bias = np.log(pos / neg)

        def build_model(hp):
            # define the keras model
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(inputSize,)))

            self.training_generator = BalancedBatchGenerator(self.X_train, self.Y_train,
                                                             batch_size=self.PARAMS['batch_size'],
                                                             sampler=RandomOverSampler(),
                                                             random_state=42)

            self.validation_generator = BalancedBatchGenerator(self.X_val, self.Y_val,
                                                               batch_size=self.PARAMS['batch_size'],
                                                               sampler=RandomOverSampler(),
                                                               random_state=42)


            for i in range(hp.Int('num_layers', 2, 8)):
                units = hp.Choice('units_' + str(i), values=[30, 36, 30, 41, 45, 60])
                deep_activation = hp.Choice('dense_activation_' + str(i), values=['relu', 'tanh'])

                model.add(Dense(units=units, activation=deep_activation))  # , kernel_initializer=initializer,))

                if self.PARAMS['Dropout']:
                    model.add(Dropout(self.PARAMS['Dropout_Rate']))

                if self.PARAMS['BatchNorm']:
                    model.add(BatchNormalization(momentum=self.PARAMS['Momentum']))

            #final_activation = hp.Choice('final_activation', values=['softmax', 'sigmoid'])
            final_activation = 'sigmoid'

            if self.PARAMS['bias_init']:
                model.add(
                    Dense(1, activation=final_activation, bias_initializer=tf.keras.initializers.Constant(value=bias)))
            else:
                model.add(Dense(1, activation=final_activation))

                # Select optimizer
                optimizer = hp.Choice('optimizer', values=['adam', 'NAdam', 'RMSprop', 'SGD'])

            lr = hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])

            # Conditional for each optimizer
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.Adam(lr, clipnorm=0.0001)

            elif optimizer == 'RMSprop':
                optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=0.0001)

            elif optimizer == 'SGD':
                optimizer = tf.keras.optimizers.SGD(lr, clipnorm=0.0001)

            elif optimizer == 'NAdam':
                optimizer = tf.keras.optimizers.Nadam(lr, clipnorm=0.0001)


            # Loss function
            if self.PARAMS['focal']:
                loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
            elif self.PARAMS['class_weights']:
                loss = weighted_binary_cross_entropy(class_weight_dict)
            else:
                loss = 'binary_crossentropy'

            # Compilation
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=['accuracy',
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()])

            return model


        if self.PARAMS['Tuner'] == 'Hyperband':
            self.tuner = Hyperband(build_model,
                                   objective=kerastuner.Objective('val_auc', direction="max"),
                                   max_epochs=self.PARAMS['epochs'],
                                   seed=1234,
                                   factor=3,
                                   overwrite=True,
                                   logger=npt_utils.NeptuneLogger(),
                                   directory=os.path.normpath('C:/'))

        elif self.PARAMS['Tuner'] == 'Bayesian':
            self.tuner = BayesianOptimization(build_model,
                                              objective=kerastuner.Objective('val_auc', direction="max"),
                                              overwrite=True,
                                              max_trials=self.PARAMS['MAX_TRIALS'],
                                              seed=1234,
                                              executions_per_trial=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                                              logger=npt_utils.NeptuneLogger(),
                                              directory=os.path.normpath('C:/'))

        elif self.PARAMS['Tuner'] == 'Random':
            self.tuner = RandomSearch(
                build_model,
                objective=kerastuner.Objective('val_auc', direction="max"),
                overwrite=True,
                seed=1234,
                max_trials=self.PARAMS['MAX_TRIALS'],
                executions_per_trial=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                logger=npt_utils.NeptuneLogger(),
                directory=os.path.normpath('C:/')
            )

        self.tuner.search(self.training_generator,
                          epochs=self.PARAMS['epochs'],
                          verbose=2,
                          validation_data=(self.validation_generator),
                          callbacks=[tf.keras.callbacks.EarlyStopping('val_auc', patience=4)])
        # Early stopping will stop epochs if val_loss doesn't improve for 4 iterations

        # self.best_model = self.hb_tuner.get_best_models(num_models=1)[0]
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        self.best_model = self.tuner.hypermodel.build(self.best_hps)

        self.history = self.best_model.fit(self.training_generator, epochs=self.PARAMS['epochs'],
                                           validation_data=(self.validation_generator), verbose=2)

        # Logs best scores, best parameters
        npt_utils.log_tuner_info(self.tuner)

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
        neptune.log_image('AUC/Epochs', auc, image_name='aucPlot')

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
        neptune.log_image('Loss/Epochs', loss, image_name='lossPlot')
        # plt.show()

        # y_predict = self.best_model.predict_classes(self.test_X) # deprecated

        y_predict = (self.best_model.predict(self.X_test) > 0.5).astype("int32")

        self.specificity = specificity_score(self.Y_test, y_predict)

        self.gmean = geometric_mean_score(self.Y_test, y_predict)

        score = self.best_model.evaluate(self.X_test, self.Y_test, verbose=0)
        self.loss = score[0]
        self.accuracy = score[1]
        self.AUC = score[4]
        self.predictedNo = y_predict.sum()
        self.trueNo = self.Y_test.sum()
        self.recall = score[3]
        self.precision = score[2]
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.Y_test, y_predict).ravel()

        neptune.log_metric('loss', self.loss)
        neptune.log_metric('accuracy', self.accuracy)
        neptune.log_metric('AUC', self.AUC)
        neptune.log_metric('specificity', self.specificity)
        neptune.log_metric('recall', self.recall)
        neptune.log_metric('precision', self.precision)
        neptune.log_metric('gmean', self.gmean)
        neptune.log_metric('True Positive', self.tp)
        neptune.log_metric('True Negative', self.tn)
        neptune.log_metric('False Positive', self.fp)
        neptune.log_metric('False Negative', self.fn)



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
            log_table('Chi2features', self.Chi2Features)

        elif self.PARAMS['Feature_Selection'] == 'MI':
            log_table('MIFeatures', self.MIFeatures)

        mins = (time.time() - self.start_time) / 60  # Time in seconds
        neptune.log_metric('minutes', mins)

class NoGen(fullNN):
    def __init__(self, PARAMS, name=None):

        self.PARAMS = PARAMS
        self.name = name

    def hpTuning(self):
        self.start_time = time.time()

        tf.keras.backend.clear_session()
        inputSize = self.X_train.shape[1]

        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.Y_train), self.Y_train)
        class_weight_dict = dict(enumerate(class_weights))
        pos = class_weight_dict[1]
        neg = class_weight_dict[0]

        bias = np.log(pos / neg)

        scalar = len(self.Y_train)
        #class_weight_dict[0] = scalar / self.Y_train.value_counts()[0]
        #class_weight_dict[1] = scalar / self.Y_train.value_counts()[1]


        weight_for_0 = (1 / self.Y_train.value_counts()[0]) * (scalar) / 2.0
        weight_for_1 = (1 / self.Y_train.value_counts()[1]) * (scalar) / 2.0
        class_weight_dict = {0: weight_for_0, 1: weight_for_1}



        def build_model(hp):
            # define the keras model
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(inputSize,)))

            for i in range(hp.Int('num_layers', 2, 8)):
                units = hp.Choice('units_' + str(i), values=[30, 36, 30, 41, 45, 60])
                deep_activation = hp.Choice('dense_activation_' + str(i), values=['relu', 'tanh'])
                model.add(Dense(units=units, activation=deep_activation))  # , kernel_initializer=initializer,))

                if self.PARAMS['Dropout']:
                    model.add(Dropout(self.PARAMS['Dropout_Rate']))

                if self.PARAMS['BatchNorm']:
                    model.add(BatchNormalization(momentum=self.PARAMS['Momentum']))

            if self.PARAMS['bias_init']:
                model.add(
                    Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(value=bias)))
            elif not self.PARAMS['bias_init']:
                model.add(Dense(1, activation='sigmoid'))

            # Select optimizer
            optimizer = hp.Choice('optimizer', values=['adam', 'NAdam', 'RMSprop', 'SGD'])

            lr = hp.Choice('learning_rate', [1e-3, 1e-4])

            # Conditional for each optimizer
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.Adam(lr, clipnorm=0.0001)

            elif optimizer == 'RMSprop':
                optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=0.0001)

            elif optimizer == 'SGD':
                optimizer = tf.keras.optimizers.SGD(lr, clipnorm=0.0001)

            elif optimizer == 'NAdam':
                optimizer = tf.keras.optimizers.Nadam(lr, clipnorm=0.0001)

            # Loss Function
            if self.PARAMS['focal']:
                loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
            elif self.PARAMS['class_weights']:
                loss = weighted_binary_cross_entropy(class_weight_dict)
            else:
                loss = 'binary_crossentropy'


            # Compilation
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=['accuracy',
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC(),
                                   tf.keras.metrics.AUC(curve='PR')])

            return model

        #batches = [32, 64, 128, 256]
        #batches = [32, 64, 128, 256, 2048, 4096, 8192]
        #batches = [128, 256, 2048, 4096]

        # Tuners don't tune batch_size, need to subclass in order to change that.
        # Can also tune epoch size, but since Hyperband has inbuilt methods for that,
        # it isn't advisable in that tuner.

        class MyBayes(kerastuner.tuners.BayesianOptimization):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`

                #kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyBayes, self).run_trial(trial, *args, **kwargs)

        class MyHB(kerastuner.tuners.Hyperband):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                #kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                super(MyHB, self).run_trial(trial, *args, **kwargs)

        class MyRand(kerastuner.tuners.RandomSearch):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                #kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyRand, self).run_trial(trial, *args, **kwargs)

        if self.PARAMS['Tuner'] == 'Hyperband':
            self.tuner = MyHB(build_model,
                              objective=kerastuner.Objective('val_auc', direction="max"),
                              max_epochs=self.PARAMS['epochs'],
                              hyperband_iterations=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                              seed=1234,
                              factor=3,
                              overwrite=True,
                              logger=npt_utils.NeptuneLogger(),
                              directory=os.path.normpath('C:/'))

        elif self.PARAMS['Tuner'] == 'Bayesian':
            self.tuner = MyBayes(build_model,
                                 objective=kerastuner.Objective('val_auc', direction="max"),
                                 overwrite=True,
                                 max_trials=self.PARAMS['MAX_TRIALS'],
                                 seed=1234,
                                 executions_per_trial=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                                 logger=npt_utils.NeptuneLogger(),
                                 directory=os.path.normpath('C:/'))

        elif self.PARAMS['Tuner'] == 'Random':
            self.tuner = MyRand(
                build_model,
                objective=kerastuner.Objective('val_auc', direction="max"),
                overwrite=True,
                seed=1234,
                max_trials=self.PARAMS['MAX_TRIALS'],
                executions_per_trial=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                logger=npt_utils.NeptuneLogger(),
                directory=os.path.normpath('C:/')
            )

        self.tuner.search(self.X_train, self.Y_train,
                          epochs=self.PARAMS['epochs'],
                          batch_size=self.PARAMS['batch_size'],
                          verbose=2,
                          validation_data=(self.X_val, self.Y_val),
                          callbacks=[tf.keras.callbacks.EarlyStopping('val_auc', patience=10)])
        # Early stopping will stop epochs if val_loss doesn't improve for 4 iterations

        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        self.best_model = self.tuner.hypermodel.build(self.best_hps)

        self.history = self.best_model.fit(self.X_train, self.Y_train, batch_size=self.PARAMS['batch_size'],
                                           epochs=self.PARAMS['epochs'],
                                           validation_data=(self.X_val, self.Y_val), verbose=2)

        # Logs best scores, best parameters
        npt_utils.log_tuner_info(self.tuner)


if __name__ == "__main__":

    outliers = ['iso','lof','ocsvm','ee']
    for o in outliers:
        def reset_random_seeds():
            os.environ['PYTHONHASHSEED'] = str(1)
            tf.random.set_seed(1)
            np.random.seed(1)
            random.seed(1)


        reset_random_seeds()

        PARAMS = {'batch_size': 8192,
                  'bias_init': False,
                  'estimator': "BayesianRidge",
                  'epochs': 30,
                  'focal': False,
                  'alpha': 0.89,
                  'gamma': 0.25,
                  'class_weights': True,
                  'initializer': 'RandomUniform',
                  'Dropout': True,
                  'Dropout_Rate': 0.20,
                  'BatchNorm': False,
                  'Momentum': 0.60,
                  'Normalize': 'MinMax',
                  'OutlierRemove': o,
                  'Feature_Selection': 'None',
                  'Feature_Num': 30,
                  'Generator': False,
                  'Tuner': "Hyperband",
                  'EXECUTIONS_PER_TRIAL': 1,
                  'MAX_TRIALS': 200,
                  'TestSplit': 0.10,
                  'ValSplit': 0.10}

        neptune.init(project_qualified_name='rachellb/MOMITuner', api_token=api_)
        neptune.create_experiment(name='MOMI Full', params=PARAMS, send_hardware_metrics=True,
                                  tags=['Weighted', 'OHE', 'FS then encode', 'Predict Mild'],
                                  description='Standardize and then Normalize')


        if PARAMS['Generator'] == False:
            model = NoGen(PARAMS, name="MOMI")
        else:
            model = fullNN(PARAMS, name="MOMI")


        # Get data
        parent = os.path.dirname(os.getcwd())
        dataPath = os.path.join(parent, 'Preprocess/momiMildPE_061821.csv')
        data = model.prepData(data=dataPath)
        model.splitData()
        data = model.imputeData()
        model.detectOutliers()
        model.scaleData()
        #features = model.featureSelection()
        model.encodeData()
        model.hpTuning()
        model.evaluateModel()

