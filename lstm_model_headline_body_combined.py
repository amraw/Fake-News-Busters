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
import sys
import io
from scorer import print_result


GLOVE_DIR = "gloVe"
RESULT_FILE = 'prediction/lstm_headline_body_combine'
TEST_FILE = 'fnc-1-master/test_stances.csv'
OBJECT_DUMP = 'objects'
EMBEDDING_DIM = 100
MAX_HEADLINE_LENGTH = 50

def lstm_model_headline_body_combin(body_length, numb_epoch):
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

    # Remove Stop words #
    test_headlines_cl = fexc.remove_stop_words_list(test_headlines_cl)
    test_bodies_cl = fexc.remove_stop_words_list(test_bodies_cl)

    # Set the tokenizer
    alltext = train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl
    token = Tokenizer(num_words=30000)
    token.fit_on_texts(alltext)
    print('Number of Unique words: ' + str(len(token.word_index.keys())))

    # Combine the headline and bodies of training data
    train_data = fexc.combine_heading_body(train_headlines_cl, train_bodies_cl)
    word_index = token.word_index

    # Converting train data to sequence
    train_data = token.texts_to_sequences(train_data)

    # Padding train data
    train_data = pad_sequences(train_data, maxlen=(MAX_HEADLINE_LENGTH+int(body_length)))

    # Converting the labels to one hot encoder
    onehotencoder = OneHotEncoder()
    train_stances_in = onehotencoder.fit_transform(train_stances_in).toarray()

    # Converting the labels to one hot encoder
    train_data, val_data, train_stances_final, stances_val = \
        train_test_split(train_data, train_stances_in, test_size=0.2, random_state=42)

    # Combining test data
    test_data = fexc.combine_heading_body(test_headlines_cl, test_bodies_cl)

    # Converting test data to sequence
    test_data = token.texts_to_sequences(test_data)

    # Padding test data
    test_data = pad_sequences(test_data, maxlen=MAX_HEADLINE_LENGTH+int(body_length))

    # Getting embedding index
    embeddings_index = models.get_embeddings_index(GLOVE_DIR)

    print('Found %s word vectors.' % len(embeddings_index))

    # Getting embedding matrix
    embedding_matrix = models.get_embedding_matrix(embedding_dim=EMBEDDING_DIM, embeddings_index=embeddings_index,
                                                   word_index=word_index)

    # Building the Model
    fake_nn = models.lstm_with_combine_headline_body(headline_length=MAX_HEADLINE_LENGTH,
                                  body_length=int(body_length),
                                  embedding_dim=EMBEDDING_DIM, word_index=word_index, embedding_matrix=embedding_matrix,
                                  activation='relu',
                                  drop_out=0.5, numb_layers=300, cells=200)

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = 'Fake_news_nlp.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    # Fitting the model
    fake_hist = fake_nn.fit(train_data, train_stances_final,
                            batch_size=128,
                            epochs=int(numb_epoch), shuffle=True,
                            validation_data=(val_data, stances_val),
                            callbacks=[early_stopping, model_checkpoint])

    # Storing the training and validation accuracy and loss in file for plot
    lstm_data = []
    with open(os.path.join(OBJECT_DUMP, "lstm_headline_body_combine" + str(body_length) + ".txt"), 'wb') as bow_hist:
        lstm_data.append(fake_hist.history['acc'])
        lstm_data.append(fake_hist.history['val_acc'])
        lstm_data.append(fake_hist.history['loss'])
        lstm_data.append(fake_hist.history['val_loss'])
        pickle.dump(lstm_data, bow_hist)

    # Predict the labels for test data
    result = fake_nn.predict([test_data], batch_size=128)

    # Store the results in the result file
    result_str = fexc.convert_lable_string(result)
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
    lstm_model_headline_body_combin(body_trunc_length, numb_epochs)