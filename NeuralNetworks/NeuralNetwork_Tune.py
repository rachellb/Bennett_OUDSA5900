#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For handling data
from datetime import datetime
import os

from sklearn.utils import class_weight

# For imputing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# For feature selection
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
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
import neptunecontrib.monitoring.kerastuner as npt_utils

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
from secret import api_
from Cleaning.Clean import *





# Initialize the project
neptune.init(project_qualified_name='rachellb/Comparisons', api_token=api_)

def weighted_loss_persample(weights, batchSize):
    def loss(y_true, y_pred):
        # The masks for the true and false values

        idx_1 = y_true == 1.
        idx_0 = y_true == 0.

        pred_1 = tf.boolean_mask(y_pred, idx_1)
        pred_1 = tf.expand_dims(pred_1, 1)

        true_1 = tf.boolean_mask(y_true, idx_1)
        true_1 = tf.expand_dims(true_1, 1)

        pred_0 = tf.boolean_mask(y_pred, idx_0)
        pred_0 = tf.expand_dims(pred_0, 1)

        true_0 = tf.boolean_mask(y_true, idx_0)
        true_0 = tf.expand_dims(true_0, 1)

        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        # The losses for the 0 classes
        losses_0 = bce(pred_0, true_0)
        losses_0 = tf.reduce_sum(losses_0, 0)  # Add back up
        losses_0 = losses_0 * weights[0]

        # The losses for the 1 classes
        losses_1 = bce(pred_1, true_1)
        losses_1 = tf.reduce_sum(losses_1, 0)  # Add back up
        losses_1 = losses_1 * weights[1]

        # Add them back up and divide by batch size
        sum = losses_0 + losses_1
        total = sum / batchSize
        return total

    return loss

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

    def prepData(self, age, data):

        self.age = age

        self.data = pd.read_csv(data)

    def normalizeData(self, data):

        scaler = MinMaxScaler()
        data_imputed = scaler.fit_transform(data)
        X_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)
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

    def splitData(self, testSize, valSize):
        self.split1=5
        self.split2=107
        X = self.data.drop(columns='Label')
        Y = self.data['Label']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, stratify=Y, test_size=testSize,
                                                                                random_state=self.split1)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              stratify=self.Y_train, test_size=valSize,
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

    def featureSelection(self, numFeatures, method):

        # If there are less features than the number selected
        numFeatures = min(numFeatures, (self.X_train.shape[1]))
        self.method = method

        if method == 1:
            model = XGBClassifier()
            model.fit(self.X_train, self.Y_train)

            # Save graph
            ax = plot_importance(model, max_num_features=numFeatures)
            fig1 = pyplot.gcf()
            #pyplot.show()
            fig1.savefig(self.dataset + 'XGBoostTopFeatures.png', bbox_inches='tight')

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

    def hpTuning(self, topFeatures):
        self.start_time = time.time()

        tf.keras.backend.clear_session()

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

            for i in range(hp.Int('num_layers', 2, 8)):
                units = hp.Choice('units_' + str(i), values=[30, 36, 30, 41, 45, 60])
                deep_activation = hp.Choice('dense_activation_' + str(i), values=['relu', 'tanh'])

                model.add(Dense(units=units, activation=deep_activation))  # , kernel_initializer=initializer,))

                if self.PARAMS['Dropout']:
                    model.add(Dropout(self.PARAMS['Dropout_Rate']))

                if self.PARAMS['BatchNorm']:
                    model.add(BatchNormalization(momentum=self.PARAMS['Momentum']))

            final_activation = hp.Choice('final_activation', values=['softmax', 'sigmoid'])

            if self.PARAMS['bias_init']:
                model.add(Dense(1, activation=final_activation))
            elif self.PARAMS['bias_init']:
                model.add(
                    Dense(1, activation=final_activation, bias_initializer=tf.keras.initializers.Constant(value=bias)))

            # Select optimizer
            optimizer = hp.Choice('optimizer', values=['adam', 'RMSprop', 'SGD'])

            lr = hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])

            # Conditional for each optimizer
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.Adam(lr, clipnorm=0.0001)

            elif optimizer == 'RMSprop':
                optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=0.0001)

            elif optimizer == 'SGD':
                optimizer = tf.keras.optimizers.SGD(lr, clipnorm=0.0001)

            if self.PARAMS['focal']:
                loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
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

        batches = [32, 64, 128, 256]

        # Tuners don't tune batch_size, need to subclass in order to change that.
        class MyBayes(kerastuner.tuners.BayesianOptimization):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[2048, 4096, 8192])
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyBayes, self).run_trial(trial, *args, **kwargs)

        class MyHB(kerastuner.tuners.Hyperband):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[2048, 4096, 8192])
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyHB, self).run_trial(trial, *args, **kwargs)

        class MyRand(kerastuner.tuners.Hyperband):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyRand, self).run_trial(trial, *args, **kwargs)

        if self.PARAMS['Tuner'] == 'Hyperband':
            self.tuner = MyHB(build_model,
                                   objective=kerastuner.Objective('val_auc', direction="max"),
                                   max_epochs=self.PARAMS['epochs'],
                                   seed=1234,
                                   factor=3,
                                   overwrite=True,
                                   logger=npt_utils.NeptuneLogger())

        elif self.PARAMS['Tuner'] == 'Bayesian':
            self.tuner = MyBayes(build_model,
                                              objective=kerastuner.Objective('val_auc', direction="max"),
                                              overwrite=True,
                                              max_trials=self.PARAMS['MAX_TRIALS'],
                                              seed=1234,
                                              executions_per_trial=2,
                                              logger=npt_utils.NeptuneLogger())

        elif self.PARAMS['Tuner'] == 'Random':
            self.tuner = MyRand(
                build_model,
                objective=kerastuner.Objective('val_auc', direction="max"),
                overwrite=True,
                seed=1234,
                max_trials=self.PARAMS['MAX_TRIALS'],
                executions_per_trial=self.PARAMS['EXECUTION_PER_TRIAL'],
                logger=npt_utils.NeptuneLogger()
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
        if self.method == 1:
            image = Image.open(self.dataset + 'XGBoostTopFeatures.png')
            neptune.log_image('XGBFeatures', image, image_name='XGBFeatures')

        elif self.method == 2:
            log_table('Chi2features', self.Chi2features)

        elif self.method == 3:
            log_table('MIFeatures', self.MIFeatures)

        mins = (time.time() - self.start_time) / 60  # Time in seconds

        neptune.log_metric('minutes', mins)

class NoGen(fullNN):
    def __init__(self, PARAMS):

        self.PARAMS = PARAMS

    def hpTuning(self, topFeatures):
        self.start_time = time.time()

        tf.keras.backend.clear_session()

        # Set all to numpy arrays
        self.X_train = self.X_train[topFeatures]
        self.Y_train = self.Y_train
        self.X_val = self.X_val[topFeatures]
        self.Y_val = self.Y_val
        self.X_test = self.X_test[topFeatures]
        self.Y_test = self.Y_test

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
                #deep_activation = 'relu'
                model.add(Dense(units=units, activation=deep_activation))  # , kernel_initializer=initializer,))

                if self.PARAMS['Dropout']:
                    model.add(Dropout(self.PARAMS['Dropout_Rate']))

                if self.PARAMS['BatchNorm']:
                    model.add(BatchNormalization(momentum=self.PARAMS['Momentum']))

            final_activation = hp.Choice('final_activation', values=['softmax', 'sigmoid'])

            if self.PARAMS['bias_init']:
                model.add(Dense(1, activation=final_activation))
            elif not self.PARAMS['bias_init']:
                model.add(
                    Dense(1, activation=final_activation, bias_initializer=tf.keras.initializers.Constant(value=bias)))

            # Select optimizer
            optimizer = hp.Choice('optimizer', values=['adam', 'RMSprop'])#, 'SGD'])

            lr = hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])

            # Conditional for each optimizer
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.Adam(lr, clipnorm=0.0001)

            elif optimizer == 'RMSprop':
                optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=0.0001)

            elif optimizer == 'SGD':
                optimizer = tf.keras.optimizers.SGD(lr, clipnorm=0.0001)

            if self.PARAMS['focal']:
                loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.PARAMS['alpha'], gamma=self.PARAMS['gamma'])
            else:
                loss = 'binary_crossentropy'

            #tfa.losses.SigmoidFocalCrossEntropy(alpha=(1/pos), gamma=0)
            #neptune.log_text('Loss Function', 'alpha=1/pos, gamma=0')
            # Compilation
            model.compile(optimizer=optimizer,
                          loss=weighted_binary_cross_entropy(class_weight_dict),
                          metrics=['accuracy',
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()])

            return model

        batches = [32, 64, 128, 256]

        # Tuners don't tune batch_size, need to subclass in order to change that.
        class MyBayes(kerastuner.tuners.BayesianOptimization):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyBayes, self).run_trial(trial, *args, **kwargs)

        class MyHB(kerastuner.tuners.Hyperband):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyHB, self).run_trial(trial, *args, **kwargs)

        class MyRand(kerastuner.tuners.RandomSearch):
            def run_trial(self, trial, *args, **kwargs):
                # You can add additional HyperParameters for preprocessing and custom training loops
                # via overriding `run_trial`
                kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=batches)
                # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
                super(MyRand, self).run_trial(trial, *args, **kwargs)

        #neptune.log_text('Tuner', 'words')

        if self.PARAMS['Tuner'] == 'Hyperband':
            self.tuner = MyHB(build_model,
                                   objective=kerastuner.Objective('val_auc', direction="max"),
                                   max_epochs=self.PARAMS['epochs'],
                                   hyperband_iterations=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                                   seed=1234,
                                   factor=3,
                                   overwrite=True,
                                   logger=npt_utils.NeptuneLogger())

        elif self.PARAMS['Tuner'] == 'Bayesian':
            self.tuner = MyBayes(build_model,
                                 objective=kerastuner.Objective('val_auc', direction="max"),
                                 overwrite=True,
                                 max_trials=self.PARAMS['MAX_TRIALS'],
                                 seed=1234,
                                 executions_per_trial=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                                 logger=npt_utils.NeptuneLogger())

        elif self.PARAMS['Tuner'] == 'Random':
            self.tuner = MyRand(
                build_model,
                objective=kerastuner.Objective('val_auc', direction="max"),
                overwrite=True,
                seed=1234,
                max_trials=self.PARAMS['MAX_TRIALS'],
                executions_per_trial=self.PARAMS['EXECUTIONS_PER_TRIAL'],
                logger=npt_utils.NeptuneLogger()
            )

        self.tuner.search(self.X_train, self.Y_train,
                          epochs=self.PARAMS['epochs'],
                          batch_size=self.PARAMS['batch_size'],
                          verbose=2,
                          validation_data=(self.X_val, self.Y_val),
                          callbacks=[tf.keras.callbacks.EarlyStopping('val_auc', patience=4)])
        # Early stopping will stop epochs if val_loss doesn't improve for 4 iterations

        # self.best_model = self.hb_tuner.get_best_models(num_models=1)[0]
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        self.best_model = self.tuner.hypermodel.build(self.best_hps)
        self.history = self.best_model.fit(self.X_train, self.Y_train, batch_size=self.PARAMS['batch_size'],
                                           epochs=self.PARAMS['epochs'],
                                           validation_data=(self.X_val, self.Y_val), verbose=2)

        # Logs best scores, best parameters
        npt_utils.log_tuner_info(self.tuner)


if __name__ == "__main__":

    PARAMS = {'batch_size': 4096,
              'bias_init': False,
              'epochs': 20,
              'focal': False,
              'alpha': 0.5,
              'gamma': 1.25,
              'class_weights': True,
              'initializer': 'RandomUniform',
              'Dropout': True,
              'Dropout_Rate': 0.20,
              'BatchNorm': True,
              'Momentum': 0.60,
              'Generator': False,
              'Tuner': "Random",
              'EXECUTIONS_PER_TRIAL': 5,
              'MAX_TRIALS': 10}

    neptune.create_experiment(name='Spect', params=PARAMS, send_hardware_metrics=True,
                              tags=['Weights = (1/classSize) * (1/len)*2'],
                              description='Comparing datasets')

    #neptune.log_text('my_text_data', 'text I keep track of, like query or tokenized word')

    if PARAMS['Generator'] == False:
        model = NoGen(PARAMS)

    else:
        model = fullNN(PARAMS)

    """
    # Get data
    #parent = os.path.dirname(os.getcwd())
    #dataPath = os.path.join(parent, 'Data/Processed/Texas/Full/Outliers/Complete/Chi2_Categorical.csv')

    model.prepData(age='Categorical',
                           data=dataPath)
    """
    data = cleanSpect()

    data = model.normalizeData(data)
    model.imputeData(data)
    model.splitData(testSize=0.10, valSize=0.10)
    features = model.featureSelection(numFeatures=20, method=2)
    model.hpTuning(features)
    model.evaluateModel()

