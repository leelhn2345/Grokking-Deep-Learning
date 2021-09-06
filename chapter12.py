# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Download & Preprocess the IMDB Dataset

# %%
# Download reviews.txt and labels.txt from here: https://github.com/udacity/deep-learning/tree/master/sentiment-network

import math
import random
from collections import Counter
import numpy as np
import sys

# %% [markdown]
# # Let's Train it!

# %%

f = open('tasksv11/en/qa1_single-supporting-fact_train.txt', 'r')
raw = f.readlines()
f.close()

tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\n", "").split(" ")[1:])

print(tokens[0:3])


# %%
vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i


def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# %%
np.random.seed(1)
embed_size = 10

# word embeddings
embed = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1

# embedding -> embedding (initially the identity matrix)
recurrent = np.eye(embed_size)

# sentence embedding for empty sentence
start = np.zeros(embed_size)

# embedding -> output weights
decoder = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1

# one hot lookups (for loss function)
one_hot = np.eye(len(vocab))

# %% [markdown]
# # Forward Propagation with Arbitrary Length

# %%


def predict(sent):

    layers = list()
    layer = {}
    layer['hidden'] = start
    layers.append(layer)

    loss = 0

    # forward propagate
    preds = list()
    for target_i in range(len(sent)):

        layer = {}

        # try to predict the next term
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))

        loss += -np.log(layer['pred'][sent[target_i]])

        # generate the next hidden state
        layer['hidden'] = layers[-1]['hidden'].dot(
            recurrent) + embed[sent[target_i]]
        layers.append(layer)

    return layers, loss

# %% [markdown]
# # Backpropagation with Arbitrary Length


# %%
# forward
for iter in range(30000):
    alpha = 0.001
    sent = words2indices(tokens[iter % len(tokens)][1:])
    layers, loss = predict(sent)

    # back propagate
    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]
        target = sent[layer_idx-1]

        if(layer_idx > 0):  # if not the first layer
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = layer['output_delta'].dot(decoder.transpose())

            # if the last layer - don't pull from a later one becasue it doesn't exist
            if(layer_idx == len(layers)-1):
                layer['hidden_delta'] = new_hidden_delta
            else:
                layer['hidden_delta'] = new_hidden_delta + \
                    layers[layer_idx +
                           1]['hidden_delta'].dot(recurrent.transpose())
        else:  # if the first layer
            layer['hidden_delta'] = layers[layer_idx +
                                           1]['hidden_delta'].dot(recurrent.transpose())

# %% [markdown]
# # Weight Update with Arbitrary Length

# %%
# forward
for iter in range(30000):
    alpha = 0.001
    sent = words2indices(tokens[iter % len(tokens)][1:])

    layers, loss = predict(sent)

    # back propagate
    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]
        target = sent[layer_idx-1]

        if(layer_idx > 0):
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = layer['output_delta'].dot(decoder.transpose())

            # if the last layer - don't pull from a
            # later one becasue it doesn't exist
            if(layer_idx == len(layers)-1):
                layer['hidden_delta'] = new_hidden_delta
            else:
                layer['hidden_delta'] = new_hidden_delta + \
                    layers[layer_idx +
                           1]['hidden_delta'].dot(recurrent.transpose())
        else:
            layer['hidden_delta'] = layers[layer_idx +
                                           1]['hidden_delta'].dot(recurrent.transpose())

    # update weights
    start -= layers[0]['hidden_delta'] * alpha / float(len(sent))
    for layer_idx, layer in enumerate(layers[1:]):

        decoder -= np.outer(layers[layer_idx]['hidden'],
                            layer['output_delta']) * alpha / float(len(sent))

        embed_idx = sent[layer_idx]
        embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * \
            alpha / float(len(sent))
        recurrent -= np.outer(layers[layer_idx]['hidden'],
                              layer['hidden_delta']) * alpha / float(len(sent))

    if(iter % 1000 == 0):
        print("Perplexity:" + str(np.exp(loss/len(sent))))

# %% [markdown]
# # Execution and Output Analysis

# %%
sent_index = 4

l, _ = predict(words2indices(tokens[sent_index]))

print(tokens[sent_index])

for i, each_layer in enumerate(l[1:-1]):
    input = tokens[sent_index][i]
    true = tokens[sent_index][i+1]
    pred = vocab[each_layer['pred'].argmax()]
    print("Prev Input:" + input + (' ' * (12 - len(input))) +
          "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)


# %%
