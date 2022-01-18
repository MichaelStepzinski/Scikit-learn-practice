# Author:   Michael Stepzinski
# Date:     8 December, 2021
# Purpose:  CS422 Project 5 Scikit-learn testing - utilities

import numpy as np
import os

def generate_vocab(dir, min_count, max_files):

    # handle directories
    posdir = dir + '/pos/'
    negdir = dir + '/neg/'
    posfilenames = os.listdir(posdir)
    negfilenames = os.listdir(negdir)
    
    # if max_files is -1, then get every file
    if max_files == -1:
        file_count = len(posfilenames) + len(negfilenames)
    else:
        # otherwise file_count = max_files
        # get as many files as max_files / 2, split evenly between pos and neg
        posfilenames = posfilenames[:int(max_files/2)]
        negfilenames = negfilenames[:int(max_files/2)]

    # create corpus dictionary
    corpus = {}

    # better name: add to corpus
    # given filenames, their directory, and corpus, add words to corpus
    generate_corpus(posfilenames, posdir, corpus)
    generate_corpus(negfilenames, negdir, corpus)

    # using corpus, get all words that occur at least min_count times
    vocab = [k for k,v in corpus.items() if v >= min_count]
    
    return vocab

def create_word_vector(fname, vocab):
    # exclude these characters for data processing
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~0123456789'
    
    # create word vector
    word_vector = np.zeros((1, len(vocab)))

    # read one file
    with open(fname, 'r') as data:
        # process data by making it lowercase, remove nums and punctuation
        temp = data.read()
        temp = temp.lower()
        for s in exclude:
            temp = temp.replace(s, '')
        temp = temp.split()
        # if processed word is a vocab word, add it to word vector
        for word in temp:
            if word in vocab:
                # add word counts to word vector
                index = vocab.index(word)
                word_vector[0][index] += 1

    return word_vector

def load_data(dir, vocab, max_files):

    # handle directories
    posdir = dir + '/pos/'
    negdir = dir + '/neg/'
    posfilenames = os.listdir(posdir)
    negfilenames = os.listdir(negdir)

    # prepend the directory to each filename to get list of paths
    posfilenames = [f'{posdir}{i}' for i in posfilenames]
    negfilenames = [f'{negdir}{i}' for i in negfilenames]
    
    # if max_files is -1, then get every file
    if max_files == -1:
        file_count = len(posfilenames) + len(negfilenames)
    else:
        # file_count = max_files
        # pos samples are half the dataset, neg are half
        posfilenames = posfilenames[:int(max_files/2)]
        negfilenames = negfilenames[:int(max_files/2)]
        # create filenames, the compilation of both
        # neg appended to pos
        filenames = posfilenames + negfilenames
        file_count = len(filenames)

    # np array of X and Y data with shapes (file_count, vocab length) and (file_count,1)
    X_data = np.zeros(shape=(file_count, len(vocab)))
    Y_data = np.zeros(shape=(file_count))

    # first half of Y is always positive reviews
    # last half of Y is always negative reviews
    # even when max_value -1 is used
    Y_data[:int(file_count/2)] = 1

    # for file_count files
    #  generate vector and add to master dataset at index num
    #  in dataset, positive example is 1, negative is 0
    for num, filename in zip(range(file_count), filenames):
        X_data[num] = create_word_vector(filename, vocab)

    return X_data, Y_data

# generate vocab helper function          
def generate_corpus(filenames, dir, corpus):

    # punctuation and numbers exclusion list
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~0123456789'
    
    # for all files, read data, process it
    #  if word in corpus, increment value by 1
    #  if not in corpus, add it,
    for filename in filenames:
        with open(dir + filename, 'r') as data:
            # read data and process
            temp = data.read()
            temp = temp.lower()
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
