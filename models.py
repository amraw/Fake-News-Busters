import torch
from torch import nn
from torch.autograd import Variable
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import os
import numpy as np
import pickle


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding1 = nn.Embedding(input_size, hidden_size)
        self.embedding2 = nn.Embedding(input_size, hidden_size)

        self.gru1 = nn.GRU(hidden_size, hidden_size, n_layers)
        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers)

        self.fc = nn.Linear(2 * hidden_size, output_size)
        # self.smax = nn.LogSoftmax(dim=2)

    def forward(self, input1, input2):
        # input1 - headline
        # input2 - body

        batch_size = input1.size(0)

        input1 = input1.t()
        input2 = input2.t()

        # print("input size ", input.size())

        embedded1 = self.embedding1(input1)
        embedded2 = self.embedding1(input2)

        # print("embedded size ", embedded.size())

        hidden1 = self.init_hidden(batch_size)
        hidden2 = self.init_hidden(batch_size)

        print("embedded1 size ", embedded1.size())
        print("embedded2 size ", embedded2.size())

        output1, hidden1 = self.gru1(embedded1, hidden1)
        output2, hidden2 = self.gru1(embedded2, hidden2)

        # print("gru output  size ", hidden.size())
        combo_hidden = torch.cat((hidden1, hidden2), 2)

        print("fc input  size ", combo_hidden.size())

        out = self.fc(combo_hidden)
        # softout = self.smax(out)

        print("fc output  size ", out.size())

        return out

    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        return result


def get_batches(name, training_reviews, training_labels, batch_size, trunc_len=100):
    s = Variable(torch.LongTensor(1, trunc_len))
    # trunc_len = 100
    file_name_x = "../features/batches."+name+".x"+".pytorch"
    file_name_y = "../features/batches." + name + ".y" + ".pytorch"
    for i in tqdm(range(len(training_reviews))):
        result = Variable(torch.LongTensor(training_reviews[i]).view(-1, 1))
        temp = result.view(-1)
        if (temp.shape[0] < trunc_len):
            pad_length = trunc_len - temp.shape[0]
            temp1 = torch.LongTensor(pad_length)
            temp1 = Variable(torch.zeros_like(temp1))
            temp1 = torch.cat((temp1, temp))
        else:
            temp1 = temp[0:trunc_len]
        s = torch.cat((s, temp1.view(1, -1)), 0)
    indices = Variable(torch.LongTensor(list(range(1, len(training_reviews) + 1))))
    s = s.index_select(0, indices)

    no_batches = len(training_labels) / batch_size
    training_x = []
    training_y = []

    for i in range(0, int(no_batches)):
        indices = Variable(torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size))))
        batch_x = s.index_select(0, indices)
        batch_y = training_labels.index_select(0, indices)
        training_x.append(batch_x)
        training_y.append(batch_y)
    return training_x, training_y


def train(input_variable1, input_variable2, encoder):
    output = encoder(input_variable1, input_variable2)
    output = torch.squeeze(output)

    return output


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))