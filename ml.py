# Author:   Michael Stepzinski
# Date:     8 December, 2021
# Purpose:  CS422 Project 5 Scikit-learn testing - ml

from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings # this is needed to stop MLP classifier from outputting warnings
from sklearn.exceptions import ConvergenceWarning 
from sklearn import svm

def dt_train(X,Y):
    # use entropy to keep tools using class methods
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X, Y)
    return clf

def kmeans_train(X):
    # clusters are 2 here
    return KMeans(n_clusters=2, random_state=0).fit(X)

def knn_train(X,Y,K):
    neigh = KNeighborsClassifier(n_neighbors=K)
    return neigh.fit(X, Y)

def perceptron_train(X,Y):
    clf = Perceptron(random_state=0)
    return clf.fit(X, Y)

@ignore_warnings(category=ConvergenceWarning)
def nn_train(X,Y,hls):
    # MLP NN has 200 epochs here
    clf = MLPClassifier(hls, random_state=0, max_iter=200)
    return clf.fit(X,Y)

def pca_train(X,K):
    pca = PCA(n_components=K)
    return pca.fit(X)

def pca_transform(X,pca):
    return pca.transform(X)

def svm_train(X,Y,k):
    clf = svm.SVC(kernel=k)
    return clf.fit(X, Y)

def model_test(X,model):
    return model.predict(X)

def compute_F1(Y,Y_hat):
    return f1_score(Y, Y_hat)
