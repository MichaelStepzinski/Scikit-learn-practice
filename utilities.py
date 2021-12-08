# Author:   Michael Stepzinski
# Date:     28 November, 2021
# Purpose:  CS422 Project 5 Scikit-learn testing - utilities

from operator import pos
import numpy as np
import sklearn as sk
#from sklearn.datasets.load_files
#from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
import random

def generate_vocab(dir, min_count, max_files):
    # handle directories
    posdir = dir + 'pos/'
    negdir = dir + 'neg/'
    posfilenames = os.listdir(posdir)
    negfilenames = os.listdir(negdir)
    
    # if max_files is -1, then get every file
    if max_files == -1:
        file_count = len(posfilenames) + len(negfilenames)
    else:
        #file_count = max_files
        posfilenames = posfilenames[:int(max_files/2)]
        negfilenames = negfilenames[:int(max_files/2)]

    # create corpus dictionary
    corpus = {}

    generate_corpus(posfilenames, posdir, corpus)
    generate_corpus(negfilenames, negdir, corpus)

    # using corpus, get all words that occur more than min_count
    vocab = [k for k,v in corpus.items() if v >= min_count]
    
    return vocab


def create_word_vector(fname, vocab):
    # exclude these characters for data processing
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~0123456789'
    
    # create word vector
    word_vector = np.zeros((1, len(vocab)))

    # read one file
    with open(fname, 'r') as data:
        # process data
        temp = data.read()
        temp = temp.lower()
        for s in exclude:
            temp = temp.replace(s, '')
        temp = temp.split()
        # if temp word is vocab word add to word vector
        for word in temp:
            if word in vocab:
                # add word counts to word vector
                index = vocab.index(word)
                word_vector[0][index] += 1

    return word_vector


def load_data(dir, vocab, max_files):
    # return train_x and train_y
    # return feature vectors with classification

    # handle directories
    posdir = dir + 'pos/'
    negdir = dir + 'neg/'
    posfilenames = os.listdir(posdir)
    negfilenames = os.listdir(negdir)

    posfilenames = [f'{posdir}{i}' for i in posfilenames]
    negfilenames = [f'{negdir}{i}' for i in negfilenames]
    
    # if max_files is -1, then get every file
    if max_files == -1:
        file_count = len(posfilenames) + len(negfilenames)
    else:
        #file_count = max_files
        posfilenames = posfilenames[:int(max_files/2)]
        negfilenames = negfilenames[:int(max_files/2)]
        filenames = posfilenames + negfilenames
        file_count = len(filenames)

    X_data = np.zeros(shape=(file_count, len(vocab)))
    Y_data = np.zeros(shape=(file_count))
    # first half of Y is always positive reviews
    # last half of Y is always negative reviews
    # even when max_value -1 is used
    Y_data[:int(file_count/2)] = 1
    # for the first max_files filenames,
    #  generate vector and add to master dataset
    #  in dataset, positive example is 1, negative is 0
    for num, filename in zip(range(file_count), filenames):
        X_data[num] = create_word_vector(filename, vocab)
        print(num)

    return X_data, Y_data
            
def generate_corpus(filenames, dir, corpus):
    # punctuation exclusion list
    #exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~0123456789'
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    for filename in filenames:
        with open(dir + filename, 'r') as data:
            # read data and process
            temp = data.read()
            #temp = temp.lower()
            for s in exclude:
                temp = temp.replace(s, '')
            # data is processed, add to corpus dictionary
            # value:key = word:occurrences
            temp = temp.split()
            for word in temp:
                if word in corpus:
                    corpus[word] += 1
                else:
                    corpus[word] = 1