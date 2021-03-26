# !/usr/bin/env python3
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

# For Outlier Detection
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

date = datetime.today().strftime('%m%d%y')  # For labelling purposes

# For recording results
import neptune
from neptunecontrib.api.table import log_table
from neptunecontrib.monitoring.keras import NeptuneMonitor
from secret import api_
from Cleaning.Clean import *


# I think layers is a list of the number of nodes per layer

# Testing custom layers
class biInteraction(tf.keras.layers.Layer):
    def __init__(self, input_shape, hidden_factor, layers, dropout):
        super(biInteraction, self).__init__()

        self.layers = layers
        num_layer = len(layers)
        self.dropout = dropout

        self.all_weights = {}

        self.all_weights['feature_embeddings'] = self.add_weight(name='feature_embeddings',
                                                                 shape=(input_shape, hidden_factor),
                                                                 initializer='random_normal',
                                                                 trainable=True)

        # TODO: figure out if this is right
        self.all_weights['feature_bias'] = self.add_weight(name='feature_bias',
                                                           shape=(input_shape,),
                                                           initializer='zeros',
                                                           trainable=True)

        self.all_weights['layer_0'] = self.add_weight(name='Layer_0',
                                                      shape=(hidden_factor, self.layers[0]),
                                                      initializer='random_normal',
                                                      trainable=True)

        self.all_weights['bias_0'] = self.add_weight(name='Bias_0',
                                                     shape=(1, self.layers[0]),
                                                     initializer='random_normal',
                                                     trainable=True)

        for i in range(1, num_layer):
            self.all_weights['layer_%d' % i] = self.add_weight(name='layer_%d' % i,
                                                               shape=(self.layers[i - 1], self.layers[i]),
                                                               initializer='random_normal',
                                                               trainable=True)

            self.all_weights['bias_%d' % i] = self.add_weight(name='bias_%d' % i,
                                                              shape=(1, self.layers[i]),
                                                              initializer='random_normal',
                                                              trainable=True)

        self.all_weights['prediction'] = self.add_weight(name='prediction',
                                                         shape=(self.layers[-1], 1),
                                                         initializer='random_normal',
                                                         trainable=True)

        # Just a constant initialized as zero?
        self.all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        # Store list of batchnorm layers, not sure why it needs to be initialized here
        self.batchnorm = []

        for i in range(num_layer):
            self.batchnorm.append(tf.keras.layers.BatchNormalization())

        self.activation_relu = tf.keras.layers.Activation('relu')

        self.batch_norm_layer = tf.keras.layers.BatchNormalization()

        self.batch_norm = True

    def call(self, inputs):

        # This section taken from https://github.com/hexiangnan/neural_factorization_machine/blob/master/NeuralFM.py

        # Train_features contains a list of lists. Each list is a sample, and what's in that list is which features
        # are present. Now I have to figure out how to get my data in that form.

        # _________ sum_square part _____________
        # get the summed up embeddings of features.

        nonzero_embeddings = tf.nn.embedding_lookup(self.all_weights['feature_embeddings'], inputs) # formerly self.train_features
        self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)  # None * K

        # get the element-multiplication
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # _________ square_sum part _____________
        self.squared_features_emb = tf.square(nonzero_embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # ________ FM __________
        self.FM = 0.5 * tf.math.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
        if self.batch_norm:
            self.FM = self.batch_norm_layer(self.FM)
        self.FM = tf.nn.dropout(self.FM, self.dropout)  # dropout at the bilinear interaction layer

        # ________ Deep Layers __________
        for i in range(0, len(self.layers)):
            self.FM = tf.add(tf.matmul(self.FM, self.all_weights['layer_%d' % i]),
                             self.all_weights['bias_%d' % i])  # None * layer[i] * 1

            if self.batch_norm:
                self.FM = self.batchnorm[i](self.FM)

            # Currently setting activation to all relus
            self.FM = self.activation_relu(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout)  # dropout at each Deep layer

        self.FM = tf.matmul(self.FM, self.all_weights['prediction'])  # None * 1

        # _________out _________
        Bilinear = tf.reduce_sum(self.FM, 1, keepdims=True)  # None * 1
        self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.all_weights['feature_bias'], inputs),
                                          1)  # None * 1

        Bias = self.all_weights['bias'] * tf.ones_like(self.train_labels)  # None * 1

        # Is this the final prediction? Why is there no activation?
        return tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1


class MyModel(tf.keras.Model):
    def __init__(self, param):
        super(MyModel, self).__init__()
        self.dense1 = biInteraction(input_shape=param['input_shape'], hidden_factor=param['hidden_factor'],
                                    layers=param['layers'], dropout=param['dropout'])

    def call(self, input_tensor):
        x = self.dense1(input_tensor)
        return x


def formatData(df):
    # This turns data into a series of lists - each list contains the non-zero features in each sample
    cols = df.columns
    bt = df.apply(lambda x: x > 0)
    bt = bt.apply(lambda x: list(cols[x.values]), axis=1)
    return bt



# Testing on Spect data since all binary apparently
data = cleanSpect()
X = data.drop(columns='Label')
Y = data['Label']

# How many variables total are there?
inputSize = X.shape[1]
X = formatData(X)
X = pad_sequences(X, maxlen=inputSize, padding='post')
#X = np.asarray(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.1,
                                                    random_state=12345)

param = {'layers': [30, 30, 30],
         'hidden_factor': 64,
         'dropout': 0.30,
         'input_shape': inputSize}

model = MyModel(param)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

model.fit(X_train, Y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(X_test, Y_test, batch_size=32, verbose=2)
