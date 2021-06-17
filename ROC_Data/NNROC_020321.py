
import pandas as pd
import numpy as np
import time

#For balancing batches
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight

#For tensorflop
import tensorflow as tf
import tensorflow_addons as tfa # For focal loss function
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

#For getting usable output
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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


    def buildModel(self,batchSize, epochs, alpha=None, gamma=None, biasInit=0):
        self.biasInit = biasInit
        self.batch_size=batchSize
        self.start_time = time.time()
        self.best_hps = {'num_layers': 6,
                         'dense_activation_0': 'tanh',
                         'dense_activation_1': 'relu',
                         'dense_activation_2': 'relu',
                         'dense_activation_3': 'relu',
                         'dense_activation_4': 'relu',
                         'dense_activation_5': 'relu',
                         'units_0': 30,
                         'units_1': 30,
                         'units_2': 30,
                         'units_3': 30,
                         'units_4': 30,
                         'units_5': 30,
                         'final_activation': 'sigmoid',
                         'optimizer': 'RMSprop',
                         'learning_rate': 0.001}

        self.alpha = alpha
        self.gamma = gamma
        LOG_DIR = f"{int(time.time())}"

        # Set all to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy().ravel()
        self.X_val = self.X_val.to_numpy()
        self.Y_val = self.Y_val.to_numpy().ravel()
        self.X_test = self.X_test.to_numpy()
        self.Y_test = self.Y_test.to_numpy().ravel()

        inputSize = self.X_train.shape[1]


        self.training_generator = BalancedBatchGenerator(self.X_train, self.Y_train,
                                                         batch_size=self.batch_size,
                                                         sampler=RandomOverSampler(),
                                                         random_state=42)

        self.validation_generator = BalancedBatchGenerator(self.X_val, self.Y_val,
                                                           batch_size=self.batch_size,
                                                           sampler=RandomOverSampler(),
                                                           random_state=42)

        # define the keras model
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(inputSize,)))

        # Hidden Layers
        for i in range(self.best_hps['num_layers']):
            self.model.add(
                Dense(units=self.best_hps['units_' + str(i)], activation=self.best_hps['dense_activation_' + str(i)]))
            self.model.add(Dropout(0.20))
            self.model.add(BatchNormalization(momentum=0.60))

        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.Y_train), self.Y_train)
        class_weight_dict = dict(enumerate(class_weights))
        pos = class_weight_dict[1]
        neg = class_weight_dict[0]

        bias = np.log(pos / neg)

        if biasInit == 0:
            # Final Layer
            self.model.add(Dense(1, activation=self.best_hps['final_activation']))
        elif biasInit == 1:
            # Final Layer
            self.model.add(Dense(1, activation=self.best_hps['final_activation'],
                                 bias_initializer=tf.keras.initializers.Constant(
                                     value=bias)))  # kernel_initializer=initializer,

        # Loss Function
        if alpha != None or gamma != None:
            loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
            self.loss = "focal_loss"
        else:
            loss = 'binary_crossentropy'
            self.loss = "binary-crossentropy"

        # Compilation
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.best_hps['learning_rate']),
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.AUC()])

        self.model.fit(self.training_generator, epochs=epochs, validation_data=(self.validation_generator), verbose=2, class_weight=class_weight_dict)

        y_pred_keras = self.model.predict(self.X_test).ravel()

        np.save('Predictions/Texas/nn_pred', y_pred_keras)

if __name__ == "__main__":
    PARAMS = {'batch_size': 8192,
              'bias_init': False,
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
              'Feature_Num': 90,
              'Generator': False,
              'TestSplit': 0.10,
              'ValSplit': 0.10}

    neuralnet = NeuralNetwork()
    neuralnet.prepData(data='Data/Texas/')
    neuralnet.buildModel(batchSize=2048, epochs=30, alpha=0.5, gamma=1.75, biasInit=0)


