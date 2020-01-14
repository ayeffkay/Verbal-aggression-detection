from learning.standard_classifiers import *
from learning.deep_learning_classifiers import *
from learning import *
import pandas as pd
from sklearn.externals import joblib


def build_nb(model_path):
    print("Building Naive Bayes classifier...")
    nb = naive_bayes()
    joblib.dump(nb, model_path + ".pkl")


def build_svm(kernel, degree, model_path):
    print("Building SVM classifier...")
    svm = support_vector_machines(kernel, degree)
    joblib.dump(svm, model_path + ".pkl")


def build_random_forest(n_estimators, max_depth, model_path):
    print("Building Random Forest classifier...")
    forest = random_forest(n_estimators, max_depth)
    joblib.dump(forest, model_path + ".pkl")


def build_extra_tree(n_estimators, max_depth, model_path):
    print("Building Extra Tree classifier...")
    trees = extra_trees(n_estimators, max_depth)
    joblib.dump(trees, model_path + ".pkl")


def build_logistic_regression(model_path):
    print("Building Logistic Regression classifier...")
    log = logistic_regression()
    joblib.dump(log, model_path + ".pkl")


def build_knn(n_neighbors, model_path):
    print("Building K Nearest Neighbors classifier...")
    neighbors = knn(n_neighbors)
    joblib.dump(neighbors, model_path + ".pkl")


def build_mlp(model_path, hid_layers, activation, nb_epochs):
    print("Building Multilayer Perceptron")
    mlp = mlp_classifier(hid_layers, activation, nb_epochs)
    joblib.dump(mlp, model_path + ".pkl")


def build_lstm(dataset, model_path, lstm_units, spatial_dropout,
               embedding=False, vocab_size=None, embedding_dim=None,
               embedding_matrix=None):
    print("Building Long-Short Term Memory network...")
    X, y, input_dim = load_dataset(dataset)
    if embedding_matrix:
        embedding_matrix = pd.read_pickle(embedding_matrix).values
    if embedding and embedding_matrix is None:
        vocab_size = np.max(X)
    model = lstm(input_dim, lstm_units, spatial_dropout,
                 embedding, vocab_size, embedding_dim,
                 embedding_matrix)
    model.save(model_path + ".h5")


def build_stacked_lstm(dataset, model_path, lstm_units, embedding=False,
                       vocab_size=None, embedding_dim=None,
                       embedding_matrix=None):
    print("Building Stacked LSTM network...")
    X, y, input_dim = load_dataset(dataset)
    if embedding_matrix:
        embedding_matrix = pd.read_pickle(embedding_matrix).values
    if embedding and embedding_matrix is None:
        vocab_size = np.max(X)
    model = stacked_lstm(input_dim, lstm_units, embedding, vocab_size,
                         embedding_dim, embedding_matrix)
    model.save(model_path + ".h5")


def build_cnn(dataset, model_path, nb_filters1, nb_filters2, kernel_size,
              pool_size, units, rate, embedding=False, vocab_size=None,
              embedding_dim=None, embedding_matrix=None):
    print("Building Convolutional Neural Network...")
    X, y, input_dim = load_dataset(dataset)
    if embedding_matrix:
        embedding_matrix = pd.read_pickle(embedding_matrix).values
    if embedding and embedding_matrix is None:
        vocab_size = np.max(X)
    model = cnn(input_dim, nb_filters1, nb_filters2, kernel_size,
                pool_size, units, rate, embedding, vocab_size,
                embedding_dim, embedding_matrix)
    model.save(model_path + ".h5")



