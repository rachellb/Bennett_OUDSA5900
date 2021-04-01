"""
Train the entity embeddings for every categorical variable and save them to disk.

We train a model with only entity embeddings (so no accompanying numeric
features), and optimize these for a while (also using cyclical learning
rates).

We then store these embeddings to disk so that they can be loaded later.

"""

import numpy as np
import pandas as pd

import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import Concatenate, Reshape, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers
from Cleaning.Clean import *

import pickle

print("Loading data...")


data = cleanBC()

split1=5
split2=107
X = data.drop(columns='Label')
Y = data['Label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.1,
                                                                        random_state=split1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                                      stratify=Y_train, test_size=0.1,
                                                                      random_state=split2)

print("Loading features...")

cat_features = X_train.columns





inputs = []
embeddings = []

for col in cat_features:
    # find the cardinality of each categorical column:
    cardinality = int(np.ceil(X_train[col].nunique() + 2))
    # set the embedding dimension:
    # at least 2, at most 50, otherwise cardinality//2
    embedding_dim = max(min((cardinality)//2, 50),2)
    print(f'{col}: cardinality : {cardinality} and embedding dim: {embedding_dim}')
    col_inputs = Input(shape=(1,))
    # Specify the embedding
    embedding = Embedding(cardinality, embedding_dim,
                          input_length=1, name=col+"_embed")(col_inputs)
    # Add a but of dropout to the embedding layers to regularize:
    embedding = SpatialDropout1D(0.1)(embedding)
    # Flatten out the embeddings:
    embedding = Reshape(target_shape=(embedding_dim,))(embedding)
    # Add the input shape to inputs
    inputs.append(col_inputs)
    # add the embeddings to the embeddings layer
    embeddings.append(embedding)

# paste all the embeddings together
x = Concatenate()(embeddings)
# Add some general NN layers with dropout.
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1)(x)

# Specify and compile the model:
embed_model = Model(inputs=inputs, outputs=outputs)
embed_model.compile(loss= "mean_squared_error",
                    optimizer="adam",
                    metrics=["mean_squared_error"])


def embedding_preproc(X_train, X_val, X_test, cat_cols):
    """
    return lists with data for train, val and test set.

    Only categorical data, no numeric. (as we are just building the
    categorical embeddings)
    """
    input_list_train = []
    input_list_val = []
    input_list_test = []

    # this bit seems to just append a list of lists, one list for each feature and the length of each list
    # is the number of samples. Is this just to get the right shape?
    for c in cat_cols:
        input_list_train.append(X_train[c].values)
        input_list_val.append(X_val[c].values)
        input_list_test.append(X_test[c].values)

    return input_list_train, input_list_val, input_list_test

# get the lists of data to feed into the Keras model:




X_embed_train, X_embed_val, X_embed_test = embedding_preproc(
                                                X_train, X_val, X_test,
                                                cat_features)

X_embed_train = np.asarray(X_embed_train).astype('float32')
#Y_train = np.asarray(Y_train).astype('float32')
X_embed_val = np.asarray(X_embed_val).astype('float32')
#Y_val = np.asarray(Y_val).astype('float32')
X_embed_test = np.asarray(X_embed_test).astype('float32')
#Y_test = np.asarray(Y_test).astype('float32')

# Fit the model
embed_history = embed_model.fit(X_embed_train,
                                Y_train.values,
                                validation_data = (X_embed_val, Y_val.values),
                                batch_size=1024,
                                epochs=15)


# Now copy the trained embeddings to a dict, and save the dict to disk.
embedding_dict = {}

for cat_col in cat_features:
    embedding_dict[cat_col] = embed_model.get_layer(cat_col + '_embed')\
                                         .get_weights()[0]

    print(f'{cat_col} dim: {len(embedding_dict[cat_col][0])}' )

pickle.dump(embedding_dict, open(str('embedding.dict'), "wb"))
