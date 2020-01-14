"""Matrix generation for embedding layer"""
from preprocessing import *
import preprocessing.weighting as w
import os
import pandas as pd


VECTORS = PATH + "corpus_db/vectors/"
TOKENS = PATH + "dataset_db/tokens/"
EMBEDDING = PATH + "dataset_db/embedding/"
SEQUENCES = PATH + "dataset_db/sequences/"


def get_seq(texts, labels, filename):
    ind, seq = w.to_sequence(corpus=texts, max_features=MAX_FEATURES)
    save(seq, filename, labels)
    return ind


names = ["glove", "w2v"]
for file in os.listdir(TOKENS):
    data = pd.read_pickle(TOKENS + file).values
    texts = data[:, 0]
    labels = data[:, -1]
    output_path = os.path.join(SEQUENCES, file.split(".")[0])
    ind = get_seq(texts, labels, output_path)
    for name in names:
        vector_path = VECTORS + name + "/" + file.split(".")[0].replace("_pos", "")
        vector = load_vectors(vector_path)
        matrix = w.build_embedding_matrix(ind, vector)
        output_name = EMBEDDING + name + "/" + file.split(".")[0]
        save(matrix, output_name)

