import os

from keras.layers.embeddings import Embedding
from keras.layers import Input, merge, TimeDistributed, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Flatten, Dropout, Dense
from keras.models import Model
from keras.layers import Bidirectional
import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from itertools import product
from functools import partial
import io
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_embeddings_index(glove_dir):
    embeddings_index = {}
    with io.open(os.path.join(glove_dir, 'glove.6B.50d.txt'), mode='r', encoding='utf8') as embedding:
        for line in embedding:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def get_embedding_matrix(word_index, embedding_dim, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def lstm_model(headline_length, body_length, embedding_dim, word_index, embedding_matrix, activation, numb_layers, drop_out, cells):
    headline_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                         input_length=headline_length, trainable=False)

    bodies_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                       input_length=body_length, trainable=False)

    headline_input = Input(shape=(headline_length,), dtype='int32')
    headline_embedding = headline_embedding_layer(headline_input)
    headline_nor = BatchNormalization()(headline_embedding)
    headline_lstm = LSTM(cells)(headline_nor)

    body_input = Input(shape=(body_length,), dtype='int32')
    body_embedding = bodies_embedding_layer(body_input)
    body_nor = BatchNormalization()(body_embedding)
    body_lstm = LSTM(cells)(body_nor)

    concat = concatenate([headline_lstm, body_lstm])
    normalize = BatchNormalization()(concat)

    preds = Dense(4, activation='softmax')(normalize)

    fake_nn = Model([headline_input, body_input], outputs=preds)
    print(fake_nn.summary())
    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return fake_nn


def lstm_model_2(globel_vectors, headline_length, body_length, embedding_dim, word_index, embedding_matrix, activation, numb_layers, drop_out, cells):
    headline_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                         input_length=headline_length, trainable=False)

    bodies_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                       input_length=body_length, trainable=False)

    headline_input = Input(shape=(headline_length,), dtype='int32')
    headline_embedding = headline_embedding_layer(headline_input)
    headline_nor = BatchNormalization()(headline_embedding)
    headline_lstm = LSTM(cells)(headline_nor)

    body_input = Input(shape=(body_length,), dtype='int32')
    body_embedding = bodies_embedding_layer(body_input)
    body_nor = BatchNormalization()(body_embedding)
    body_lstm = LSTM(cells)(body_nor)

    global_vector_input = Input(shape=(globel_vectors,), dtype='float32')

    concat = concatenate([headline_lstm, body_lstm, global_vector_input])
    normalize = BatchNormalization()(concat)

    preds = Dense(4, activation='softmax')(normalize)

    fake_nn = Model([headline_input, body_input, global_vector_input], outputs=preds)
    print(fake_nn.summary())
    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return fake_nn


def lstm_model_3(headline_length, body_length, embedding_dim, word_index, embedding_matrix, activation, numb_layers, drop_out, cells):
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                input_length=headline_length + body_length, trainable=False)

    input = Input(shape=(headline_length + body_length,), dtype='int32')
    embedding = embedding_layer(input)

    lstm = LSTM(cells)(embedding)

    preds = Dense(4, activation='softmax')(lstm)
    fake_nn = Model(input, outputs=preds)
    print(fake_nn.summary())
    fake_nn.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    return fake_nn


def feed_forward_network(input_vector, activation, numb_layers, drop_out):
    input = Input(shape=(input_vector,), dtype='int32')
    dense = Dense(numb_layers, activation=activation)(input)
    dropout = Dropout(drop_out)(dense)
    normalize = BatchNormalization()(dropout)
    preds = Dense(4, activation='softmax')(normalize)
    fake_nn = Model(input, outputs=preds)
    print(fake_nn.summary())
    fake_nn.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    return fake_nn
