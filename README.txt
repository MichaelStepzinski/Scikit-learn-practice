# Author:   Michael Stepzinski
# Date:     8 December, 2021
# Purpose:  CS422 Project 5 Scikit-learn testing - writeup

def generate_vocab(dir, min_count, max_files)
    Generate vocab handles directories using os, strings, and lists, 
    then gets the file count, which is the total number of files to 
    generate a vocab from. positive files = negative files, getting 
    max_files / 2 files from each. A dictionary is used to store file data 
    to have a quicker implementation. generate_corpus takes corpus, the filenames, 
    and their directories, then adds the data to corpus dictionary. Corpus' keys 
    are the words, the values are the number of occurences total. vocab, a list, 
    then takes all key:value pairs of corpus and adds them to the vocab if their 
    value lists them as occuring >= min_count times.

def generate_corpus(filenames, dir, corpus)
    Generate corpus reads data from a list of filenames, refines the data, 
    then adds these words to the corpus in key:value pairs of word:num_occurrences.

def create_word_vector(fname, vocab)
    Create word vector is given one file at a time and a vocab. 
    It generates a (1, vocab_length) array to store all data, 
    reads a single file, refines it, and adds a word to the same index of 
    the vector as the vocab list, or increases its count.

def load_data(dir, vocab, max_files)
    Load data handles directories, prepends all filenames with their respective 
    directories, then handles the file numbers by getting total files and the 
    required number from pos and neg. All pos and neg filenames (prepended with paths) 
    are added together into one list. The X and Y data variables are shaped according 
    to the dataset. Every file is passed into create_word_vector and this vector 
    is added to the X_data at the next blank index.

def dt_train(X,Y)
    DT train takes X and Y data, uses sklearn's decisionTreeClassifier in the entropy 
    mode (just like in class), fits this tree to the data, and returns the fitted model.

def kmeans_train(X)
    KMeans train takes X data, produces 2 cluster centers and returns the model.

def knn_train(X,Y,K)
    KNN Train takes K neighbors, X and Y data, and returns the model fitted to the data.

def perceptron_train(X,Y)
    Perceptron train takes X and Y data, then returns a model fitted on this data.

def nn_train(X,Y, hls)
    NN train takes X and Y data and a hidden layer size, builds a NN on this, then trains 
    based on the X and Y data, and returns this model.

def pca_train(X,K)
    PCA train takes X data and K eigenvectors to use and projects the data using those 
    eigenvectors, then returns those eigenvectors.

def pca_transform(X,pca)
    PCA transform takes X data and pca data, then projects X using the Principal Components 
    and returns that data.

def svm_train(X,Y,k)
    SVM train takes X and Y data and a kernel type, then fits a SVM to the data.

def model_test(X,model)
    Model test takes X (test) data and a model, then predicts the labels for that data using 
    the models passed in.

def compute_F1(Y, Y_hat)
    Compute F1 takes Y ground truth and Y predicted data, then uses sklearn's f1_score to 
    calculate the F1 score of the data. This calculates 2(precision*recall)/(precision+recall)
