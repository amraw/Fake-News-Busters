import os
from load_data import load_data
from preprocessing import Preprocessing
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import csv
import sys
import io
from word2index import Word2Index
from embeddings_create import get_embeddings_index, get_embedding_matrix
import torch
from torch import nn

GLOVE_DIR = "../gloVe"
PREDICTIONS_FILE = '../prediction/bi_lstm_concat_'
TEST_FILE = '../fnc-1-master/test_stances.csv'
OBJECT_DUMP = '../objects'
EMBEDDING_DIM = 50

def feed_forward_model():
    fexc = Preprocessing()
    data = load_data()
    data.set_path(path='../fnc-1-master')
    train_stance_data = data.get_headline_body_stance()
    train_bodies_data = data.get_body_id_text()
    train_headlines, train_bodies, train_stances = data.get_mapped_id_body(train_stance_data, train_bodies_data)

    # Train headlines
    train_headlines_cl = fexc.get_clean_data(train_headlines)
    train_bodies_cl = fexc.get_clean_data(train_bodies)
    train_stances_cl = fexc.get_clean_data(train_stances)
    # remove stop words
    train_headlines_cl = fexc.remove_stop_words_list(train_headlines_cl)
    train_bodies_cl = fexc.remove_stop_words_list(train_bodies_cl)

    # Word to integer
    train_stances_in = fexc.convert_lable_int(train_stances_cl)
    onehotencoder = OneHotEncoder()
    train_stances_in = onehotencoder.fit_transform(train_stances_in).toarray()
    # perform stemming
    # train_headlines_cl = fexc.perform_stemming_list(train_headlines_cl)
    # train_bodies_cl = fexc.perform_stemming_list(train_bodies_cl)

    ## Test Data Load ##
    data.set_name("test")
    test_stance_data = data.get_headline_body_stance()
    test_bodies_data = data.get_body_id_text()
    test_headlines, test_bodies = data.get_mapped_id_body(test_stance_data, test_bodies_data, data_type="test")

    test_headlines_cl = fexc.get_clean_data(test_headlines)
    test_bodies_cl = fexc.get_clean_data(test_bodies)

    # Remove Stop words #
    test_headlines_cl = fexc.remove_stop_words_list(test_headlines_cl)
    test_bodies_cl = fexc.remove_stop_words_list(test_bodies_cl)

    # Creating embedding matrix
    alltext = train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl
    word2Index = Word2Index()
    word2Index.fit(alltext)

    train_headlines_seq = word2Index.text_to_index(train_headlines_cl)
    train_bodies_seq = word2Index.text_to_index(train_bodies_cl)

    test_headlines_seq = word2Index.text_to_index(test_headlines_cl)
    test_bodies_seq = word2Index.text_to_index(test_bodies_cl)

    word_index = word2Index.get_word2idx()
    embeddings_index = get_embeddings_index(GLOVE_DIR)
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = get_embedding_matrix(embedding_dim=EMBEDDING_DIM, embeddings_index=embeddings_index,
                                            word_index=word_index)
    print("Embedding Matrix dimension:"+str(embedding_matrix.shape))

    embed = nn.Embedding((embedding_matrix.shape)[0], EMBEDDING_DIM)
    # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
    embed.weight.data.copy_(torch.from_numpy(embedding_matrix))


feed_forward_model()
