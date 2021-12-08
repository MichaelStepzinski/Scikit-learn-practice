import numpy as np
import sys
#This should run outside of your Project5 Directory
sys.path.insert(0, 'Project5')

import utilities as utils
import ml

#
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#

#preliminaries
train_dir = "aclImdb_v1/aclImdb/train/"
test_dir = "aclImdb_v1/aclImdb/test/"
max_files = 1000
min_word_count = 2
PCA_K = 50

#Generate vocab and load the data (creating feature vectors)
vocab = utils.generate_vocab(train_dir, min_word_count, max_files)
train_X, train_Y = utils.load_data(train_dir, vocab, max_files)
test_X, test_Y = utils.load_data(test_dir, vocab, max_files)

#Apply PCA to training data, apply resulting transformation to training and test data
pca = ml.pca_train(train_X, PCA_K)
proj_train_X = ml.pca_transform(train_X, pca)
proj_test_X = ml.pca_transform(test_X, pca)

#Decision Tree
dt = ml.dt_train(train_X,train_Y)
#test_predict = ml.model_test(test_X, dt)
test_predict = dt.predict(test_X)
f1 = ml.compute_F1(test_Y,test_predict)
print("Decision Tree:", f1)

dt = ml.dt_train(proj_train_X,train_Y)
test_predict = ml.model_test(proj_test_X, dt)
f1 = ml.compute_F1(test_Y,test_predict)
print("Decision Tree + PCA:", f1)

#KMeans
kmeans = ml.kmeans_train(train_X)
test_predict = ml.model_test(test_X, kmeans)
f1 = ml.compute_F1(test_Y,test_predict)
print("KMeans:", f1)

kmeans = ml.kmeans_train(proj_train_X)
test_predict = ml.model_test(proj_test_X, kmeans)
f1 = ml.compute_F1(test_Y,test_predict)
print("KMeans + PCA:", f1)

#KNN
knn = ml.knn_train(train_X,train_Y,3)
test_predict = ml.model_test(test_X, knn)
f1 = ml.compute_F1(test_Y,test_predict)
print("KNN:", f1)

knn = ml.knn_train(proj_train_X,train_Y,3)
test_predict = ml.model_test(proj_test_X, knn)
f1 = ml.compute_F1(test_Y,test_predict)
print("KNN + PCA:", f1)

#Perceptron
perceptron = ml.perceptron_train(train_X,train_Y)
test_predict = ml.model_test(test_X, perceptron)
f1 = ml.compute_F1(test_Y,test_predict)
print("Perceptron:", f1)

perceptron = ml.perceptron_train(proj_train_X,train_Y)
test_predict = ml.model_test(proj_test_X, perceptron)
f1 = ml.compute_F1(test_Y,test_predict)
print("Perceptron + PCA:", f1)

#Neural Network
nn = ml.nn_train(train_X,train_Y,(5,2))
test_predict = ml.model_test(test_X, nn)
f1 = ml.compute_F1(test_Y,test_predict)
print("Neural Network:", f1)

nn = ml.nn_train(proj_train_X,train_Y,(5,2))
test_predict = ml.model_test(proj_test_X, nn)
f1 = ml.compute_F1(test_Y,test_predict)
print("Neural Network + PCA:", f1)

#SVM
svm = ml.svm_train(train_X,train_Y,"poly")
test_predict = ml.model_test(test_X, svm)
f1 = ml.compute_F1(test_Y,test_predict)
print("SVM:", f1)

svm = ml.svm_train(proj_train_X,train_Y,"poly")
test_predict = ml.model_test(proj_test_X, svm)
f1 = ml.compute_F1(test_Y,test_predict)
print("SVM + PCA:", f1)


