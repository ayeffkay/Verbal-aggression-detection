""" Feature weighting for labeled dataset"""
from preprocessing import *
import preprocessing.weighting as w
import os
import pandas as pd

path = PATH + "dataset_db/tokens/"
weights_folder = PATH + "dataset_db/weights/"


def get_bin(texts, labels, filename):
    feat, tf_bin = w.binary(corpus=texts, max_features=MAX_FEATURES)
    save(tf_bin, weights_folder + filename + "_tf_bin", labels=labels, columns=feat)


def get_freq(texts, labels, filename):
    feat, tf_freq = w.frequency(corpus=texts, max_features=MAX_FEATURES)
    save(tf_freq, weights_folder + filename + "_tf_freq", labels=labels, columns=feat)


def get_tf_idf(texts, labels, filename):
    feat, tf_idf = w.tf_idf(corpus=texts, max_features=MAX_FEATURES)
    save(tf_idf, weights_folder + filename + "_tf_idf", labels, feat)


def get_hash(texts, labels, filename):
    hashs = w.hashing(corpus=texts, dim=MAX_FEATURES)
    save(hashs, weights_folder + filename + "_hash", labels)


for file in os.listdir(path):
    print("Processing ", file)
    data = pd.read_pickle(path + file).get_values()
    texts = data[:, 0]
    labels = data[:, 1]
    get_bin(texts, labels, file.split(".")[0])
    get_freq(texts, labels, file.split(".")[0])
    get_tf_idf(texts, labels, file.split(".")[0])
    get_hash(texts, labels, file.split(".")[0])
