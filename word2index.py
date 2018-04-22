import os
import torch

class Word2Index(object):

    def __init__(self):
        self.word2idx = {0: 0}
        self.idx2word = [0]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def fit(self, corpus):
        unique_words = set((" ".join(corpus)).split())
        for word in unique_words:
            if len(word) > 1:
                self.add_word(word)
        print("Number of Unique words: "+str(len(self.idx2word)))

    def text_to_index(self, input):
        tokenized_string = []
        for line in input:
            tokens = line.split()
            sub_result = []
            for token in tokens:
                if len(token) > 1:
                    sub_result.append(self.word2idx[token])
            tokenized_string.append(sub_result)
        return tokenized_string

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word
