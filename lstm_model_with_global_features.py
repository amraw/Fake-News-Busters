from load_data import load_data
from preprocessing import Preprocessing
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from feature_extraction import get_text_features, get_tfidf_vec
import models
import pickle
import csv
import sys
import io
from scorer import print_result


GLOVE_DIR = "gloVe"
RESULT_FILE = 'prediction/lstm_seperate_headline_body_glob_feature'
TEST_FILE = 'fnc-1-master/test_stances.csv'
OBJECT_DUMP = 'objects'
EMBEDDING_DIMENSION = 100
MAX_HEADLINE_LENGTH = 50

def lstm_model_glob_feature(body_length, numb_layers):

    prepro = Preprocessing()
    data = load_data()

    # Loading train data from files
    data.set_path(path='fnc-1-master')
    train_stance_data = data.get_headline_body_stance()
    train_bodies_data = data.get_body_id_text()
    train_headlines, train_bodies, train_stances = data.get_mapped_id_body(train_stance_data, train_bodies_data)

    # Removing punctuation and stop words from the headline and body of train data
    train_headlines_cl = prepro.get_clean_data(train_headlines)
    train_bodies_cl = prepro.get_clean_data(train_bodies)
    train_stances_cl = prepro.get_clean_data(train_stances)

    # Convert labels to integer
    train_stances_in = prepro.convert_lable_int(train_stances_cl)

    # Load the test data
    data.set_name("test")
    test_stance_data = data.get_headline_body_stance()
    test_bodies_data = data.get_body_id_text()
    test_headlines, test_bodies = data.get_mapped_id_body(test_stance_data, test_bodies_data, data_type="test")

    # Removing punctuation and stop words from the headline and body of test data
    test_headlines_cl = prepro.get_clean_data(test_headlines)
    test_bodies_cl = prepro.get_clean_data(test_bodies)

    # Get global feature
    tfidf_vec = get_tfidf_vec(train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl)

    train_global_feature = get_text_features('train', train_headlines, train_headlines_cl, train_bodies, train_bodies_cl,
                                            tfidf_vec)
    test_global_feature = get_text_features('test', test_headlines, test_headlines_cl, test_bodies, test_bodies_cl,
                                           tfidf_vec)
    global_feature_length = train_global_feature.shape[1]

    # Set the tokenizer
    total_text = train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl
    token = Tokenizer(num_words=30000)
    token.fit_on_texts(total_text)
    print('Number of Unique words: ' + len(token.word_index.keys()))

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

    # Splitting data to train and validation
    train_headlines_final, headlines_val, train_bodies_final, bodies_val, train_stances_final, stances_val, train_global, val_global = \
        train_test_split(train_headlines_seq, train_bodies_seq, train_stances_in, train_global_feature, test_size=0.2, random_state=42)

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
    fake_nn = models.lstm_model_with_featrues(globel_vectors=global_feature_length, headline_length=MAX_HEADLINE_LENGTH, body_length=int(body_length),
                                embedding_dim=EMBEDDING_DIMENSION, word_index=word_index, embedding_matrix=embedding_matrix,
                                activation='relu',
                                drop_out=0.5, numb_layers=300, cells=200)

    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = 'Fake_news_nlp.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    # Fitting the model
    fake_hist = fake_nn.fit([train_headlines_final, train_bodies_final, train_global], train_stances_final, batch_size=128,
                            epochs=int(numb_epochs), shuffle=True, validation_data=([headlines_val, bodies_val, val_global], stances_val),
                            callbacks=[early_stopping, model_checkpoint])

    # Storing the training and validation accuracy and loss in file for plot
    bow_list_data = []
    with open(os.path.join(OBJECT_DUMP, "lstm_seperate_headline_body_glob_feature_" + str(body_length) + ".txt"), 'wb') as bow_hist:
        bow_list_data.append(fake_hist.history['acc'])
        bow_list_data.append(fake_hist.history['val_acc'])
        bow_list_data.append(fake_hist.history['loss'])
        bow_list_data.append(fake_hist.history['val_loss'])
        pickle.dump(bow_list_data, bow_hist)

    # Predict the labels for test data
    result = fake_nn.predict([test_headlines_seq, test_bodies_seq, test_global_feature], batch_size=128)

    # Store the results in the result file
    result_str = prepro.convert_lable_string(result)
    with io.open(TEST_FILE, mode='r', encoding='utf8') as read_file:
        test_stance = csv.DictReader(read_file)
        with io.open(RESULT_FILE + "_" + str(body_length) + ".csv", mode='w',
                     encoding='utf8') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=['Headline', 'Body ID', 'Stance'])
            writer.writeheader()
            for sample, prediction in zip(test_stance, result_str):
                writer.writerow({'Body ID': sample['Body ID'], 'Headline': sample['Headline'], 'Stance': prediction})

        # Print the Accuracy, competition score and confusion matrix
        print_result("fnc-1-master/competition_test_stances.csv", RESULT_FILE + "_" + str(body_length) + ".csv")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Pass at least 3 arguments")
        exit(1)
    _, body_trunc_length, numb_epochs = sys.argv
    print('Truncation_length: ' + body_trunc_length)
    print('Number of epochs: ' + numb_epochs)
    lstm_model_glob_feature(body_trunc_length, numb_epochs)
