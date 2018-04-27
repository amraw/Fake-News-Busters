from load_data import load_data
from preprocessing import Preprocessing
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import models
import pickle
import csv
from scorer import print_result
import sys
import io

GLOVE_DIR = "gloVe"
PREDICTIONS_FILE = 'prediction/lstm_seperate_headline_body_'
TEST_FILE = 'fnc-1-master/test_stances.csv'
OBJECT_DUMP = 'objects'
EMBEDDING_DIMENSION = 100
MAX_HEADLINE_LENGTH = 50

def lstm_model(body_length):
    fexc = Preprocessing()
    data = load_data()

    # Loading train data from files
    data.set_path(path='fnc-1-master')
    train_stance_data = data.get_headline_body_stance()
    train_bodies_data = data.get_body_id_text()
    train_headlines, train_bodies, train_stances = data.get_mapped_id_body(train_stance_data, train_bodies_data)

    # Removing punctuation and stop words from the headline and body of train data
    train_headlines_cl = fexc.get_clean_data(train_headlines)
    train_bodies_cl = fexc.get_clean_data(train_bodies)
    train_stances_cl = fexc.get_clean_data(train_stances)

    # Convert labels to integer
    train_stances_in = fexc.convert_lable_int(train_stances_cl)

    # Load the test data
    data.set_name("test")
    test_stance_data = data.get_headline_body_stance()
    test_bodies_data = data.get_body_id_text()
    test_headlines, test_bodies = data.get_mapped_id_body(test_stance_data, test_bodies_data, data_type="test")

    # Removing punctuation and stop words from the headline and body of test data
    test_headlines_cl = fexc.get_clean_data(test_headlines)
    test_bodies_cl = fexc.get_clean_data(test_bodies)

    # Set the tokenizer
    total_text = train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl
    token = Tokenizer(num_words=30000)
    token.fit_on_texts(total_text)
    print(len(token.word_index.keys()))

    # Convert headline and body to sequence
    train_headlines_seq = token.texts_to_sequences(train_headlines_cl)
    train_bodies_seq = token.texts_to_sequences(train_bodies_cl)
    word_index = token.word_index

    # Padding the headline and body
    train_headlines_seq = pad_sequences(train_headlines_seq, maxlen=MAX_HEADLINE_LENGTH)
    train_bodies_seq = pad_sequences(train_bodies_seq, maxlen=int(body_length))

    # Converting the labels to one hot encoder
    onehotencoder = OneHotEncoder()
    train_stances_in = onehotencoder.fit_transform(train_stances_in).toarray()

    # Splitting data to test and validation
    train_headlines_final, headlines_val, train_bodies_final, bodies_val, train_stances_final, stances_val = \
        train_test_split(train_headlines_seq, train_bodies_seq, train_stances_in, test_size=0.2, random_state=42)

    # Convert headline and body to sequence
    test_headlines_seq = token.texts_to_sequences(test_headlines_cl)
    test_bodies_seq = token.texts_to_sequences(test_bodies_cl)

    # Padding the headline and body
    test_headlines_seq = pad_sequences(test_headlines_seq, maxlen=MAX_HEADLINE_LENGTH)
    test_bodies_seq = pad_sequences(test_bodies_seq, maxlen=int(body_length))

    # Getting embedding index
    embeddings_index = models.get_embeddings_index(GLOVE_DIR)

    print('Found %s word vectors.' % len(embeddings_index))

    # Getting embedding matrix
    embedding_matrix = models.get_embedding_matrix(embedding_dim=EMBEDDING_DIMENSION, embeddings_index=embeddings_index,
                                                   word_index=word_index)

    # Getting the model
    fake_nn = models.lstm_model(headline_length=MAX_HEADLINE_LENGTH, body_length=int(body_length),
                                embedding_dim=EMBEDDING_DIMENSION, word_index=word_index, embedding_matrix=embedding_matrix,
                                activation='relu',
                                drop_out=0.5, numb_layers=100, cells=200)

    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = 'Fake_news_nlp.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    # Fitting the model
    fake_hist = fake_nn.fit([train_headlines_final, train_bodies_final], train_stances_final, batch_size=128,
                            epochs=1, shuffle=True, validation_data=([headlines_val, bodies_val], stances_val),
                            callbacks=[early_stopping, model_checkpoint])

    # Storing the training and validation accuracy and loss in file for plot
    bow_list_data = []
    with open(os.path.join(OBJECT_DUMP, "lstm_seperate_headline_body_" + str(body_length) + ".txt"), 'wb') as bow_hist:
        bow_list_data.append(fake_hist.history['acc'])
        bow_list_data.append(fake_hist.history['val_acc'])
        bow_list_data.append(fake_hist.history['loss'])
        bow_list_data.append(fake_hist.history['val_loss'])
        pickle.dump(bow_list_data, bow_hist)

    # Predict the labels for test data
    result = fake_nn.predict([test_headlines_seq, test_bodies_seq], batch_size=128)

    # Store the results in the result file
    result_str = fexc.convert_lable_string(result)
    with io.open(TEST_FILE, mode='r', encoding='utf8') as read_file:
        test_stance = csv.DictReader(read_file)
        with io.open(PREDICTIONS_FILE + "_" + str(body_length) + "_" + ".csv", mode='w',
                     encoding='utf8') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=['Headline', 'Body ID', 'Stance'])
            writer.writeheader()
            for sample, prediction in zip(test_stance, result_str):
                writer.writerow({'Body ID': sample['Body ID'], 'Headline': sample['Headline'], 'Stance': prediction})

    # Print the Accuracy, competition score and confusion matrix
    print_result("fnc-1-master/competition_test_stances.csv", PREDICTIONS_FILE + "_" + str(body_length) + "_" + ".csv")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Pass at least 2 arguments")
        exit(1)
    _, body_length = sys.argv
    print(body_length)
    lstm_model(body_length)
