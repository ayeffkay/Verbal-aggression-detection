"""Tokenization, stemming and lemmatization for both datasets"""

import preprocessing.tokenization as t
from preprocessing.load_db import Dataset, Corpus
from preprocessing import PATH, save


def generate_tokens(texts):
    print("Tokenization...")
    texts = t.tokenize(corpus=texts)
    return texts


def generate_stems(texts):
    print("Stemming...")
    texts = t.stemming(corpus=texts)
    return texts


def generate_lems(texts):
    print("Lemmatization...")
    texts = t.lemmatization1(corpus=texts)
    return texts


def generate_lems_pos(texts):
    print("Lemmatization with POS...")
    texts = t.lemmatization2(corpus=texts)
    return texts


db1 = {
    "name": Corpus(),
    "folder": "corpus_db/tokens/",
    "actions": [generate_tokens, generate_stems, generate_lems]

}
db2 = {
    "name": Dataset(),
    "folder": "dataset_db/tokens/",
    "actions": [generate_tokens, generate_stems, generate_lems, generate_lems_pos]
}

params = [db1, db2]

for db in params:
    name =  db["name"]
    for action in db["actions"]:
        text = action(texts=name.texts)
        path = PATH + db["folder"] + action.__name__
        save(text, path, labels=name.labels, columns=None)
