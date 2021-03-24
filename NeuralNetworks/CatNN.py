
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


# Testing custom layers
class biInteraction(tf.keras.layers.Layer):
    def __init__(self, input_shape, hidden_factor):
        super(biInteraction, self).__init__()
        self.weights = self.add_weight(Name='feature_embeddings',
                                       Shape=(input_shape[1], hidden_factor),
                                       Initializer='random_normal',
                                       Trainable=True)

        self.bias = self.add_weight(Name='feature_bias',
                                    Shape=(input_shape[1],),
                                    Initializer='zeros',
                                    Trainable=True)

    def call(self, inputs):
        # This section taken from https://github.com/hexiangnan/neural_factorization_machine/blob/master/NeuralFM.py

        # _________ sum_square part _____________
        # get the summed up embeddings of features.
        nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
        self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)  # None * K
        # get the element-multiplication
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # _________ square_sum part _____________
        self.squared_features_emb = tf.square(nonzero_embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # ________ FM __________
        self.FM = 0.5 * tf.sub(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
        if self.batch_norm:
            self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
        self.FM = tf.nn.dropout(self.FM, self.dropout_keep[-1])  # dropout at the bilinear interactin layer

        # ________ Deep Layers __________
        for i in range(0, len(self.layers)):
            self.FM = tf.add(tf.matmul(self.FM, self.weights['layer_%d' % i]),
                             self.weights['bias_%d' % i])  # None * layer[i] * 1
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_%d' % i)  # None * layer[i] * 1
            self.FM = self.activation_function(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
        self.FM = tf.matmul(self.FM, self.weights['prediction'])  # None * 1

        # _________out _________
        Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
        self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),
                                          1)  # None * 1
        Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1

        # Is this the final prediction? Why is there no activation?
        return tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1