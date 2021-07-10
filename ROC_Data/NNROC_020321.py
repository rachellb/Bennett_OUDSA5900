
import pandas as pd
import numpy as np
import time

#For balancing batches
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight

#For tensorflow
import tensorflow as tf
import tensorflow_addons as tfa # For focal loss function
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow.keras.backend as K

#For getting usable output
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


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

class NeuralNetwork():

    def __init__(self, PARAMS):

        self.PARAMS = PARAMS

    def prepData(self, data):
        path = data

        self.X_train = pd.read_csv(path + 'X_Train.csv')
        self.Y_train = pd.read_csv(path + 'Y_Train.csv')
        self.X_test = pd.read_csv(path + 'X_Test.csv')
        self.Y_test = pd.read_csv(path + 'Y_Test.csv')
        self.X_val = pd.read_csv(path + 'X_Val.csv')
        self.Y_val = pd.read_csv(path + 'Y_Val.csv')

    def buildModel(self):

        LOG_DIR = f"{int(time.time())}"

        # Set all to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy()
        self.X_val = self.X_val.to_numpy()
        self.Y_val = self.Y_val.to_numpy()
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
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.Y_train), y=np.ravel(self.Y_train))
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

        # Question - Can you put a list in here?

        self.history = self.model.fit(self.training_generator, epochs=self.PARAMS['epochs'], verbose=2)

        y_pred_keras = self.model.predict(self.X_test).ravel()

        return y_pred_keras



class NoGen(NeuralNetwork):
    def buildModel(self):

        LOG_DIR = f"{int(time.time())}"
        """
        # Set all to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy()
        self.X_val = self.X_val.to_numpy()
        self.Y_val = self.Y_val.to_numpy()
        self.X_test = self.X_test.to_numpy()
        self.Y_test = self.Y_test.to_numpy()
        """

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
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.Y_train), y=np.ravel(self.Y_train))
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

        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.PARAMS['batch_size'],
                                      epochs=self.PARAMS['epochs'],
                                      verbose=2)

        y_pred_keras = self.model.predict(self.X_test).ravel()

        return y_pred_keras



if __name__ == "__main__":

    dataset = 'MOMI'
    method = 'CE'

    PARAMS = {'num_layers': 5,
              'dense_activation_0': 'tanh',
              'dense_activation_1': 'tanh',
              'dense_activation_2': 'tanh',
              'dense_activation_3': 'tanh',
              'dense_activation_4': 'tanh',
              'units_0': 36,
              'units_1': 30,
              'units_2': 60,
              'units_3': 41,
              'units_4': 36,
              'final_activation': 'sigmoid',
              'optimizer': 'RMSprop',
              'learning_rate': 0.001,
              'batch_size': 8192,
              'bias_init': 0,
              'epochs': 30,
              'focal': False,
              'alpha': 0.91,
              'gamma': 1.25,
              'class_weights': False,
              'initializer': 'RandomUniform',
              'Dropout': True,
              'Dropout_Rate': 0.20,
              'BatchNorm': False,
              'Momentum': 0.60,
              'Feature_Selection': 'Chi2',
              'Generator': False,
              'MAX_TRIALS': 5}

    if PARAMS['Generator'] == False:
        neuralnet = NoGen(PARAMS)
    else:
        neuralnet = NeuralNetwork(PARAMS)

    neuralnet.prepData(data='Data/' + dataset + '/070821_')
    y_pred_keras = neuralnet.buildModel()

    np.save('Predictions/' + dataset + '/nn_pred_' + method, y_pred_keras)


