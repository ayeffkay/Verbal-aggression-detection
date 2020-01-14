"""Weighting with word2vec and glove word vectors"""
from preprocessing import *
import preprocessing.weighting as w
import pandas as pd
import os


def get_average(texts, vectors, labels, filename):
    avg_wv = w.avg_wv(corpus=texts, vectors=vectors)
    save(avg_wv, filename, labels=labels)


def get_tfidf_wv(texts, vectors, labels, features, filename):
    tfidf_wv = w.tfidf_avg_wv(texts, vectors, features)
    save(tfidf_wv, filename, labels=labels)


names = ["glove", "w2v"]
vectors_path = PATH + "corpus_db/vectors/"
tokens_folder = PATH + "dataset_db/tokens/"
weigths_path = PATH + "dataset_db/weights/"
for file in os.listdir(tokens_folder):
    print("Dataset: ", file)
    data = pd.read_pickle(tokens_folder + file).get_values()
    texts = data[:, 0]
    labels = data[:, 1]
    for name in names:
        print("Vector:", name)
        path = vectors_path + name + "/" + file.split(".")[0]
        path = path.replace("_pos", "")
        vec = load_vectors(path)
        output_path = weigths_path + name + "_" + file.split(".")[0]
        get_average(texts, vec, labels, output_path)
        output_path = weigths_path + name + "_tf_idf_" + file.split(".")[0]
        get_tfidf_wv(texts, vec, labels, MAX_FEATURES, output_path)


