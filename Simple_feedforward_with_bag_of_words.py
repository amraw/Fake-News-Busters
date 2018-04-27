from load_data import load_data
from preprocessing import Preprocessing
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import models
import pickle
import csv
from feature_extraction import get_text_features, get_tfidf_vec, headline_body_vec, get_tfreq_vectorizer
import io
import sys
from scorer import print_result


GLOVE_DIR = "gloVe"
RESULT_FILE = 'prediction/feedforward_network'
TEST_FILE = 'fnc-1-master/test_stances.csv'
OBJECT_DUMP = 'objects'


def feed_forward_model(numb_epocs):
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

    # Convert labels to one hot encoder
    train_stances_in = prepro.convert_lable_int(train_stances_cl)
    onehotencoder = OneHotEncoder()
    train_stances_in = onehotencoder.fit_transform(train_stances_in).toarray()

    # Load test data
    data.set_name("test")
    test_stance_data = data.get_headline_body_stance()
    test_bodies_data = data.get_body_id_text()
    test_headlines, test_bodies = data.get_mapped_id_body(test_stance_data, test_bodies_data, data_type="test")

    # Removing punctuation and stop words from the headline and body of test data
    test_headlines_cl = prepro.get_clean_data(test_headlines)
    test_bodies_cl = prepro.get_clean_data(test_bodies)

    # Get all the text features
    tfidf_vec = get_tfidf_vec(train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl)

    train_global_feature = get_text_features('train', train_headlines, train_headlines_cl, train_bodies, train_bodies_cl,
                                            tfidf_vec)
    test_global_feature = get_text_features('test', test_headlines, test_headlines_cl, test_bodies, test_bodies_cl,
                                           tfidf_vec)

    # Headline Body vector representation
    bow_vectorizer, tfreq_vectorizer = get_tfreq_vectorizer(train_headlines_cl, train_bodies_cl , lim_unigram=5000)

    train_data = headline_body_vec('train', train_headlines_cl, train_bodies_cl, train_global_feature,
                                   bow_vectorizer, tfreq_vectorizer)

    test_data = headline_body_vec('test', test_headlines_cl, test_bodies_cl, test_global_feature,
                                  bow_vectorizer, tfreq_vectorizer)

    # Train validation split
    train_data_final, train_val, train_stances_final, stances_val = \
        train_test_split(train_data, train_stances_in, test_size=0.2, random_state=42)

    # Get the Model
    fake_nn = models.feed_forward_network(input_vector=(train_data.shape)[1], activation='relu', drop_out=0.5,
                                          numb_layers=100)

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = 'Fake_news_nlp.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    # Fitting the model
    fake_hist = fake_nn.fit([train_data_final], train_stances_final, batch_size=128,
                            epochs=int(numb_epocs), shuffle=True, validation_data=([train_val], stances_val),
                            callbacks=[early_stopping, model_checkpoint])

    # Storing the training and validation accuracy and loss in file for plot
    bow_list_data = []
    with open(os.path.join(OBJECT_DUMP, "feedforward_network" + ".txt"), 'wb') as bow_hist:
        bow_list_data.append(fake_hist.history['acc'])
        bow_list_data.append(fake_hist.history['val_acc'])
        bow_list_data.append(fake_hist.history['loss'])
        bow_list_data.append(fake_hist.history['val_loss'])
        pickle.dump(bow_list_data, bow_hist)

    # Predict the labels for test data
    result = fake_nn.predict([test_data], batch_size=128)

    # Store the results in the result file
    result_str = prepro.convert_lable_string(result)
    with io.open(TEST_FILE, mode='r', encoding='utf8') as read_file:
        test_stance = csv.DictReader(read_file)
        with io.open(RESULT_FILE + "feedforward_network" + ".csv", mode='w',
                     encoding='utf8') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=['Headline', 'Body ID', 'Stance'])
            writer.writeheader()
            for sample, prediction in zip(test_stance, result_str):
                writer.writerow({'Body ID': sample['Body ID'], 'Headline': sample['Headline'], 'Stance': prediction})

    # Print the Accuracy, competition score and confusion matrix
    print_result("fnc-1-master/competition_test_stances.csv",
                 RESULT_FILE + "feedforward_network" + ".csv")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Pass at least 2 arguments")
        exit(1)
    _, numb_epochs = sys.argv
    feed_forward_model(numb_epocs=numb_epochs)