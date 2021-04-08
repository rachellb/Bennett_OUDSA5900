import os
import matplotlib.pylab as plt
import pandas as pd

import datetime, warnings, scipy
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Activation, Reshape
from tensorflow.keras.layers import Concatenate, Dropout
from tensorflow.keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')

from Cleaning.Clean import *


data = cleanBC()

# From this example https://github.com/mmortazavi/EntityEmbedding-Working_Example/blob/master/EntityEmbedding.ipynb

features = ['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape',
                    'MarinalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin',
                    'NormalNucleoli', 'Mitosis']

target = ['Label']

data.dropna(inplace=True)

X = data.drop(columns='Label')
Y = data['Label']


X_train, X_test, y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.1, random_state=1234)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=1234)
"""
embed_cols = [i for i in X_train.select_dtypes(include=['object'])]

for i in embed_cols:
    print(i, data[i].nunique())

embed_cols = [i for i in X_train.select_dtypes(include=['object'])]


# converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        # For each column, raw values stores the unique values that appear
        raw_vals = np.unique(X_train[c])
        val_map = {}
        # For each unique value that appears in the column:
        for i in range(len(raw_vals)):

            val_map[raw_vals[i]] = i
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test


for categorical_var in X_train.select_dtypes(include=['object']):
    cat_emb_name = categorical_var.replace(" ", "") + '_Embedding'

    no_of_unique_cat = X_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat) / 2), 50))

    print('Categorical Variable:', categorical_var,
          'Unique Categories:', no_of_unique_cat,
          'Embedding Size:', embedding_size)


for categorical_var in X_train.select_dtypes(include=['object']):
    input_name = 'Input_' + categorical_var.replace(" ", "")
    print(input_name)

input_models = []
output_embeddings = []
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

for categorical_var in X_train.select_dtypes(include=['object']):
    # Name of the categorical variable that will be used in the Keras Embedding layer
    cat_emb_name = categorical_var.replace(" ", "") + '_Embedding'

    # Define the embedding_size
    no_of_unique_cat = X_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat) / 2), 50))

    # One Embedding Layer for each categorical variable
    input_model = Input(shape=(1,))
    output_model = Embedding(no_of_unique_cat, embedding_size)(input_model)
    output_model = Reshape(target_shape=(embedding_size,))(output_model)

    # Appending all the categorical inputs
    input_models.append(input_model)

    # Appending all the embeddings
    output_embeddings.append(output_model)

# Other non-categorical data columns (numerical).
# I define single another network for the other columns and add them to our models list.
input_numeric = Input(shape=(len(X_train.select_dtypes(include=numerics).columns.tolist()),))
embedding_numeric = Dense(128)(input_numeric)
input_models.append(input_numeric)
output_embeddings.append(embedding_numeric)

# At the end we concatenate altogther and add other Dense layers
output = Concatenate()(output_embeddings)
output = Dense(1000, kernel_initializer="uniform")(output)
output = Activation('relu')(output)
output = Dropout(0.4)(output)
output = Dense(512, kernel_initializer="uniform")(output)
output = Activation('relu')(output)
output = Dropout(0.3)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(inputs=input_models, outputs=output)
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mse', 'mape'])

X_train_list, X_val_list, X_test_list = preproc(X_train, X_val, X_test)

X_train_list = np.asarray(X_train_list)
y_train = np.asarray(y_train)
X_val_list = np.asarray(X_val_list)

history  =  model.fit(X_train_list,y_train,validation_data=(X_val_list,y_val) , epochs =  1000 , batch_size = 512, verbose= 2)

"""

categorical_columns = X_train.columns
input_train_list = []
input_test_list = []
for c in categorical_columns:
  input_train_list.append(X_train[c].values)
  input_test_list.append(X_test[c].values)


input_models = []
# output embeddings will capture all the output embeddings
output_embeddings = []


for c in categorical_columns:
  cat_emb_name = c + '_Embedding'
  # Identifying the number of unique values in the category
  no_of_unique_cat = X_train[c].nunique()
  # Defining the output embedding size - currently taken it to be 10 for
  # simplicity, Jeremy Howard Fast AI course gives empirical formula for this to
  # be int(min(np.ceil(no_of_unique_cat/2, 50))) i.e choosing the minimum of
  # 50 or half the number of categories in the column
  embedding_size = 10
  input_model = Input(shape=(1,), name=c + '_Input')
  output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(input_model)
  output_model = Reshape(target_shape=(embedding_size,))(output_model)

  input_models.append(input_model)
  output_embeddings.append(output_model)

output = Concatenate()(output_embeddings)
output = Dense(512, kernel_initializer="uniform")(output)
output = Activation('relu')(output)
output = Dropout(0.4)(output)
output = Dense(256, kernel_initializer="uniform")(output)
output = Activation('relu')(output)
output = Dropout(0.3)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(inputs=input_models, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


rt=tf.ragged.constant(input_train_list)
input_train_list = tf.data.Dataset.from_tensor_slices(rt)

rt=tf.ragged.constant(input_test_list)
input_test_list = tf.data.Dataset.from_tensor_slices(rt)

rt=tf.ragged.constant(y_train)
y_train = tf.data.Dataset.from_tensor_slices(rt)

rt=tf.ragged.constant(Y_test)
Y_test = tf.data.Dataset.from_tensor_slices(rt)



history  =  model.fit(input_train_list, y_train, validation_data=(input_test_list, Y_test) , epochs =  10 , batch_size = 32, verbose= 2)