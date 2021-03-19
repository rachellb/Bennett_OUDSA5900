
import pandas as pd
import numpy as np
import os

#For model
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.utils import class_weight
import tensorflow_addons as tfa
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler

# For stratified Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold

# For Auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from imblearn.metrics import geometric_mean_score


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

    def prepData(self, data):

        data = pd.read_csv(data)

        X = data.drop(columns='Preeclampsia/Eclampsia')
        Y = data['Preeclampsia/Eclampsia']

        return X, Y

    def setData(self, X_train, X_test, Y_train, Y_test):

        self.X_train = X_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.Y_train = Y_train.to_numpy()
        self.Y_test = Y_test.to_numpy().ravel()

    def buildModel(self):

        # Set all to numpy arrays
        self.X_train = self.X_train
        self.Y_train = self.Y_train
        self.X_test = self.X_test
        self.Y_test = self.Y_test

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
        """
        # Reset class weights for use in loss function
        scalar = len(self.Y_train)
        class_weight_dict[0] = scalar / self.Y_train.value_counts()[0]
        class_weight_dict[1] = scalar / self.Y_train.value_counts()[1]
        """

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

        self.model.fit(self.training_generator,
                                      epochs=self.PARAMS['epochs'],
                                      verbose=2)

        self.predictions = (self.model.predict(self.X_test) > 0.5).astype("int32")


    def evaluateModel(self):

        fpr, tpr, thresholds = roc_curve(self.Y_test, self.predictions)
        auc_ = metrics.auc(fpr, tpr)
        gmean = geometric_mean_score(self.Y_test, self.predictions)


        return auc_, gmean

class NoGen(fullNN):
    def buildModel(self):
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

        pos = self.Y_train.sum()
        neg = len(self.Y_train) - pos
        bias = np.log(pos / neg)

        if self.PARAMS['bias_init'] == 0:
            # Final Layer
            self.model.add(Dense(1, activation=self.PARAMS['final_activation']))

        elif self.PARAMS['bias_init'] == 1:
            # Final Layer
            self.model.add(Dense(1, activation=self.PARAMS['final_activation'],
                                 bias_initializer=tf.keras.initializers.Constant(
                                     value=bias)))
        """
        # Reset class weights for use in loss function
        scalar = len(self.Y_train)
        class_weight_dict[0] = scalar / self.Y_train.value_counts()[0]
        class_weight_dict[1] = scalar / self.Y_train.value_counts()[1]
        """

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

        self.model.fit(self.X_train, self.Y_train, batch_size=self.PARAMS['batch_size'],
                                      epochs=self.PARAMS['epochs'],
                                      verbose=2, class_weight=class_weight_dict)

        self.predictions = (self.model.predict(self.X_test) > 0.5).astype("int32")




if __name__ == "__main__":

    PARAMS = {'num_layers': 3,
              'dense_activation_0': 'tanh',
              'dense_activation_1': 'relu',
              'dense_activation_2': 'relu',
              'units_0': 30,
              'units_1': 36,
              'units_2': 45,
              'final_activation': 'sigmoid',
              'optimizer': 'Adam',
              'learning_rate': 0.001,
              'batch_size': 4096,
              'bias_init': 0,
              'epochs': 30,
              'focal': False,
              'alpha': 0.5,
              'gamma': 1.25,
              'class_weights': False,
              'initializer': 'RandomUniform',
              'Dropout': True,
              'Dropout_Rate': 0.20,
              'BatchNorm': False,
              'Momentum': 0.60,
              'Generator': False,
              'MAX_TRIALS': 5}

    parent = os.path.dirname(os.getcwd())

    name = 'Texas'
    weight = PARAMS['class_weights']

    if name == 'Oklahoma':
        path = os.path.join(parent, 'Data/Processed/Oklahoma/Complete/Full/Outliers/Chi2_Categorical.csv')
    else:
        path = os.path.join(parent, 'Data/Processed/Texas/Full/Outliers/Complete/Chi2_Categorical.csv')


    if weight == False:
        model = fullNN(PARAMS)

    else:
        model = NoGen(PARAMS)

    X, y = model.prepData(data=path)

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)

    aucList = []
    gmeanList = []

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.setData(X_train, X_test, y_train, y_test)

        model.buildModel()

        auc, gmean = model.evaluateModel()

        aucList.append(auc)
        gmeanList.append(gmean)


    if weight:
        np.save('AUC/' + name + '/CSDNN_auc', aucList)
        np.save('Gmean/' + name + '/CSDNN_gmean', gmeanList)
    else:
        np.save('AUC/' + name + '/DNN_auc', aucList)
        np.save('Gmean/' + name + '/DNN_gmean', gmeanList)

