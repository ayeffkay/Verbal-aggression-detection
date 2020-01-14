"""Divide all words by whitespaces (for glove learning)"""
from preprocessing import PATH
import os
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence

path = PATH + "corpus_db/tokens/"


def save(file, output_name):
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(file)


for file in os.listdir(path):
    if file.endswith(".pkl"):
        texts = pd.read_pickle(path+file).get_values()[:, 0]
        whitespaced = ""
        for text in texts:
            text = text_to_word_sequence(text)
            whitespaced += " ".join(text) + " "
        save(file=whitespaced, output_name=path + file.split(".")[0])



