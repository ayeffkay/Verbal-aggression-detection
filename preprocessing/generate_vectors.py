"""Word2vec and glove learning"""
import preprocessing.weighting as w
from preprocessing import *
from keras.preprocessing.text import text_to_word_sequence
import os
import pandas as pd


path = PATH + "corpus_db/tokens/"
vectors_path = PATH + "corpus_db/vectors/"
dim = 64
window = 10
sample = 1E-3
min_ct = 7
epochs = 15


def get_vectors():
    for file in os.listdir(path):
        s = "Processing file {}...".format(file)
        print(s)
        if file.endswith(".pkl"):
            data = pd.read_pickle(path + file).get_values()[:, 0]
            sentences = [text_to_word_sequence(d) for d in data]
            vectors = w.train_w2v(sentences, size=dim,
                                   window=window,
                                   sample=sample,
                                   min_ct=min_ct,
                                   epochs=epochs)
            name = "/w2v/"
        else:
            vectors = w.train_glove(corpus_name=path + file,
                                    size=dim,
                                    window=window,
                                    min_ct=min_ct,
                                    epochs=epochs)
            name = "/glove/"
        output_name = vectors_path + name + file.split(".")[0]
        print("Saving vectors...")
        save_vectors(vectors, output_name)


get_vectors()


