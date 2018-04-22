import os
from load_data import load_data
from preprocessing import Preprocessing
import os
import numpy as np
#from data_analysis import DataAnalysis
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import models
import pickle
import csv
from feature_extraction import get_all_features, get_tfidf_vec, headline_body_vec, get_tfreq_vectorizer
import tensorflow as tf
import keras.backend as K
import sys
import io


GLOVE_DIR = "../gloVe"
PREDICTIONS_FILE = '../prediction/feedforward_network'
TEST_FILE = '../fnc-1-master/test_stances.csv'
OBJECT_DUMP = '../objects'


def feed_forward_model(numb_layers):
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
    #train_headlines_cl = fexc.remove_stop_words_list(train_headlines_cl)
    #train_bodies_cl = fexc.remove_stop_words_list(train_bodies_cl)

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
    #test_headlines_cl = fexc.remove_stop_words_list(test_headlines_cl)
    #test_bodies_cl = fexc.remove_stop_words_list(test_bodies_cl)

    # Get global feature
    tfidf_vec = get_tfidf_vec(train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl)

    train_global_feature = get_all_features('train', train_headlines, train_headlines_cl, train_bodies, train_bodies_cl,
                                            tfidf_vec)
    test_global_feature = get_all_features('test', test_headlines, test_headlines_cl, test_bodies, test_bodies_cl,
                                           tfidf_vec)

    # Headline Body vector

    bow_vectorizer, tfreq_vectorizer = get_tfreq_vectorizer(train_headlines_cl, train_bodies_cl , lim_unigram=5000)

    train_data = headline_body_vec('train', train_headlines_cl, train_bodies_cl, train_global_feature,
                                    bow_vectorizer, tfreq_vectorizer)

    test_data = headline_body_vec('test', test_headlines_cl, test_bodies_cl, test_global_feature,
                                    bow_vectorizer, tfreq_vectorizer)

    train_data_final, train_val, train_stances_final, stances_val = \
        train_test_split(train_data, train_stances_in, test_size=0.2, random_state=42)

    fake_nn = models.feed_forward_network(input_vector=(train_data.shape)[1], activation='relu', drop_out=0.5,
                                          numb_layers=100)
    print(fake_nn.summary())
    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = 'Fake_news_nlp.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    fake_hist = fake_nn.fit([train_data_final], train_stances_final, batch_size=128,
                            epochs=1, shuffle=True, validation_data=([train_val], stances_val),
                            callbacks=[early_stopping, model_checkpoint])
    bow_list_data = []
    with open(os.path.join(OBJECT_DUMP, "feedforward_network_" + str(numb_layers) + ".txt"), 'wb') as bow_hist:
        bow_list_data.append(fake_hist.history['acc'])
        bow_list_data.append(fake_hist.history['val_acc'])
        bow_list_data.append(fake_hist.history['loss'])
        bow_list_data.append(fake_hist.history['val_loss'])
        pickle.dump(bow_list_data, bow_hist)

    result = fake_nn.predict([test_data], batch_size=128)

    result_str = fexc.convert_lable_string(result)
    with io.open(TEST_FILE, mode='r', encoding='utf8') as read_file:
        test_stance = csv.DictReader(read_file)
        with io.open(PREDICTIONS_FILE + "feedforward_network_"  + str(numb_layers) + ".csv", mode='w',
                     encoding='utf8') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=['Headline', 'Body ID', 'Stance'])
            writer.writeheader()
            for sample, prediction in zip(test_stance, result_str):
                writer.writerow({'Body ID': sample['Body ID'], 'Headline': sample['Headline'], 'Stance': prediction})

    """word2Index = Word2Index()
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
    #print(embedding_matrix[0:3, :])
    batch_size = 256
    no_batches = int(len(train_stances_in) / batch_size)
    training_labels = Variable(torch.LongTensor(train_stances_in))
    headline_seq_length = 50
    body_seq_length = 150
    training_x, training_z = get_batches('headline', train_headlines_seq, training_labels, batch_size, headline_seq_length)
    training_y, training_z = get_batches('body', train_bodies_seq, training_labels, batch_size, body_seq_length)

    hidden_size = 256
    n_classes = 4

    encoder = EncoderRNN(embedding_matrix.shape[0], hidden_size, n_classes)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print_every = 50
    n_iters = no_batches
    plot_every = 100

    train_no_batches = (no_batches)
    no_epochs = 20
    for e in range(0, no_epochs):
        for iter in range(1, train_no_batches + 1):

            output = train(training_x[iter - 1], training_y[iter - 1], encoder)
            print(output.shape,  training_z[iter - 1].shape)
            print(output[55, :])
            print(training_z[iter - 1][55])
            loss = criterion(output, training_z[iter - 1])
            encoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()

            print_loss_total += loss
            plot_loss_total += loss
            # print(iter)
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)"""

feed_forward_model(1)
