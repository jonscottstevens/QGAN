# work in progress:  a GAN for forging and detecting English questions

import json
import string
from math import floor

# load SQuAD training data (https://rajpurkar.github.io/SQuAD-explorer/)
data = json.load(open('train-v1.1.json', 'rb'))['data']

# extract and preprocess  the questions

Qs = [data[n]['paragraphs'][0]['qas'][0]['question'] for n in range(0,len(data))] # extract
Qs = list(filter(None, [s.translate(str.maketrans('', '', string.punctuation)).split(" ") for s in Qs])) # remove punct
Qs = [[w.lower() for w in q] for q in Qs]  # convert to lower

maxlen = 12  # the max length of question we'll train on

def fix_length(q):
    return q[0:maxlen] if len(q) >= maxlen else q + ([" "] * (maxlen - len(q))) # truncates/pads seq length to maxlen

Qs = [["#"] + q + ["?"] for q in Qs]  # add start and stop symbols (# begins a question and ? ends it)
Qs = [q for q in Qs if len(q)<=maxlen]  # learn shorter Qs only
Qs = [fix_length(q) for q in Qs]  # fix length of each Q sequence

# numerify and then one-hot encode the data

vocab = list(set([w for q in Qs for w in q]))
get_vocab = dict(zip(range(0, len(vocab)), vocab))
get_index = {v: k for k, v in get_vocab.items()}

def translate(q):
    return [get_index[w] for w in q]

def detranslate(q):
    return [get_vocab[w] for w in q]

Qs = [translate(q) for q in Qs]

def one_hot(i):
    vec = [0]*len(vocab)
    vec[i] = 1
    return(vec)

# now the GAN part

import torch

def encode(q):
    return [one_hot(i) for i in q]

encoded = [list(map(list, zip(*encode(q)))) for q in Qs]

question_shape = maxlen * len(vocab)  # the dimensions of an utterance

import torch.nn as nn
import torch.nn.functional as F

seed_size = 100  # the seed is a random sequence of numbers for random sentence generation

# the generator is a two-layer feed-forward network with random input
class Generator(nn.Module):
    def __init__(self, first_size, second_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(seed_size, first_size)
        self.map2 = nn.Linear(first_size, second_size)
        self.map3 = nn.Linear(second_size, question_shape)

    def forward(self, x):
        x = nn.Dropout(p=0.5)(F.elu(self.map1(x)))
        x = nn.Dropout(p=0.5)(F.elu(self.map2(x)))
        return F.sigmoid(self.map3(x)).view(1, len(vocab), maxlen)  # reshape output for processing by discriminator

# the discriminator is a three-channel convolutional network
class Discriminator(nn.Module):
    def __init__(self, hidden_size, kernel_size_a=5, kernel_size_b=6, kernel_size_c=9, pool_a=2, pool_b=2, pool_c=2):
        super(Discriminator, self).__init__()
        self.pool_a, self.pool_b, self.pool_c = pool_a, pool_b, pool_c
        self.map1a = nn.Conv1d(len(vocab), hidden_size, kernel_size_a)  # conv layer, first channel
        self.map1b = nn.Conv1d(len(vocab), hidden_size, kernel_size_b)  # conv layer, second channel
        self.map1c = nn.Conv1d(len(vocab), hidden_size, kernel_size_c)  # conv layer, third channel
        self.dim2a = floor((maxlen - kernel_size_a + 1) / pool_a) * hidden_size  # size of output of map1a
        self.dim2b = floor((maxlen - kernel_size_b + 1) / pool_b) * hidden_size  # size of output of map1b
        self.dim2c = floor((maxlen - kernel_size_c + 1) / pool_c) * hidden_size  # size of output of map1c
        self.map2 = nn.Linear(self.dim2a + self.dim2b + self.dim2c, 1)  # merged layer

    def forward(self, x):
        x1 = F.elu(self.map1a(x))  # exponential linear activation function
        x1 = torch.nn.MaxPool1d(self.pool_a)(x1)  # insert a pooling layer
        x1 = torch.nn.Dropout(0.5)(x1)  # insert a dropout layer
        x2 = F.elu(self.map1b(x))  # second channel
        x2 = torch.nn.MaxPool1d(self.pool_b)(x2)
        x2 = torch.nn.Dropout(0.5)(x2)
        x3 = F.elu(self.map1c(x))  # third channel
        x3 = torch.nn.MaxPool1d(self.pool_c)(x3)
        x3 = torch.nn.Dropout(0.5)(x3)
        x = torch.cat((x1, x2, x3), 2)  # here is where the channels are merged
        x = x.view(x.numel())  # flatten output to feed into the sigmoid layer
        return F.sigmoid(self.map2(x)).view(1)  # output a score between 0 and 1 (0=fake; 1=real)

# get ready to train

num_epochs = 100

D = Discriminator(hidden_size=50)
G = Generator(first_size=500, second_size=5000)

import torch.optim as optim

d_optimizer = optim.Adam(D.parameters(), lr=0.02, betas=(0.9, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999))
criterion = nn.BCELoss()  # binary cross entropy

def generate_seed():
    return torch.normal(mean=torch.zeros(seed_size), std=torch.ones(seed_size)) # generate random input to generator

# now we are ready to train

for n in range(0, num_epochs):
    D.zero_grad()

    # train discriminator on training (authentic) data with ones as labels
    for e in encoded:
        d_authentic = torch.FloatTensor([e])
        d_authentic_predictions = D(d_authentic)
        d_authentic_error = criterion(d_authentic_predictions, torch.ones(1))
        d_authentic_error.backward()  # cache gradients

    #  train discriminator on generated (forged) data with zeros as labels
    for i in range(0, len(encoded)):
        seed = torch.FloatTensor(generate_seed())
        d_forged = G(seed).detach()
        d_forged_predictions = D(d_forged)
        d_forged_error = criterion(d_forged_predictions, torch.zeros(1))
        d_forged_error.backward()  # cache gradients
    d_optimizer.step()  # update discriminator weights

    # prepare to train generator
    G.zero_grad()
    g_steps = 100

    # train generator with discriminator success as loss function
    for i in range(0, g_steps):
        seed = torch.FloatTensor(generate_seed())
        g_forgeries = G(seed)
        d_forgery_detection = D(g_forgeries)
        g_error = criterion(d_forgery_detection, torch.ones(1))
        g_error.backward()  # cache gradients
    g_optimizer.step()  # update generator weights

    # generate 100 random sentences after each round of training; print any sentences containing start and stop symbols
    for n in range(0,100):
        seed = torch.FloatTensor(generate_seed())
        prediction = G(seed).transpose(1, 2)
        maxes = [int(w.max(0)[1]) for w in prediction[0]]
        words = [get_vocab[w] for w in maxes]
        sentence = " ".join(words)
        if "#" in sentence and "?" in sentence:
            out = sentence.split("#")[1].split("?")[0]
            if out:
                print("# " + out + " ?")
    print("################")
