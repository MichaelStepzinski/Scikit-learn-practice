# Author:   Michael Stepzinski
# Date:     28 November, 2021
# Purpose:  CS422 Project 5 Scikit-learn testing - utilities

import numpy as np
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
import random

def generate_vocab(dir, min_count, max_files):
    # handle directories
    posdir = dir + '/pos/'
    negdir = dir + '/neg/'
    posfilenames = os.listdir(posdir)
    negfilenames = os.listdir(negdir)
    
    # if max_fiels is -1, then get every file
    if max_files == -1:
        file_count = len(posfilenames) + len(negfilenames)
    else:
        file_count = max_files

    # create corpus containing each string
    corpus = []

    # for filename in pos, add data to corpus
    for filename in posfilenames:
        with open(posdir + filename, 'r') as data:
            corpus.append(data.read())
        if len(corpus) == int(file_count/2):
            break
    for filename in negfilenames:
        with open(negdir + filename, 'r') as data:
            corpus.append(data.read())
        if len(corpus) == file_count:
            break
    # use sklearn to vectorize corpus
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    # get vectorized words from corpus
    words = vectorizer.get_feature_names_out()
    # get count of each vectorized word in corpus
    counts = X.toarray()

    # declare final vocab list
    vocab = []

    # if word occurs min_count times, add to vocabulary
    sums = counts.sum(axis=0)
    for word, count in zip(words, sums):
        if count >= min_count:
            vocab.append(word)
    return vocab

def create_word_vector(fname, vocab):
    # read one file
    with open(fname, 'r') as data:
        # raw is in list format to work with vectorizer.fit_transform
        raw = [data.read()]

    # vectorize this file
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(raw)
    # get vectorized words from file
    file_words = vectorizer.get_feature_names_out()
    # get count of each vectorized word in file
    counts = X.toarray()
    # since raw was in list format, it gets interpreted as 2D array
    # this is a workaround to get everything working
    counts = counts[0]
    # make counts a list to be input into a dictionary
    counts = counts.tolist()

    # occurences per word in file
    file_word_dictionary = dict(zip(file_words, counts))

    # occurences per word in vocab
    word_dictionary = dict.fromkeys(vocab, '0')

    # each time word in vocab occurs, increment spot in word_vector
    for word in file_word_dictionary:
        if word in vocab:
            # increase occurence count
            word_dictionary[word] = file_word_dictionary[word]
    word_vector = np.fromiter(word_dictionary.values(), dtype=int)

    return word_vector


def load_data(dir, vocab, max_files):
    # return train_x and train_y
    # return feature vectors with classification

    # handle directories
    posdir = dir + '/pos/'
    negdir = dir + '/neg/'
    posfilenames = os.listdir(posdir)
    negfilenames = os.listdir(negdir)
    
    # if max_files is -1, then get every file
    if max_files == -1:
        file_count = len(posfilenames) + len(negfilenames)
    else:
        file_count = max_files

    X_data = np.zeros(shape=(file_count, len(vocab)))
    Y_data = np.zeros(shape=(file_count))
    Y_data[:int(file_count/2)] = 1
    # for the first max_files filenames,
    #  generate vector and add to master dataset
    #  in dataset, positive example is 1, negative is 0
    for num, filename in zip(range(int(file_count/2)), posfilenames):
        X_data[num] = create_word_vector(posdir + filename, vocab)
    for num, filename in zip(range(int(file_count/2)), negfilenames):
        X_data[num] = create_word_vector(negdir + filename, vocab)

    return X_data, Y_data
            