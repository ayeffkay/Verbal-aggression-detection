import pandas as pd
import json

# POS for pymorphy2
pos1 = {"NOUN", "ADJF", "ADJS",
        "VERB", "INF", "NPRO"}
# части речи для mystem
pos2 = ["Grammeme.Substantive",
        "Grammeme.Adjective",
        "Grammeme.Verb",
        "Grammeme.SubstPronoun",
        "Grammeme.AdjPronoun"]
# tags to ignore
constr1 = {"Abbr", "Name", "Surn", "Patr",
           "Geox", "Orgn", "Trad"}
constr2 = ["Grammeme.FirstName",
           "Grammeme.Patr",
           "Grammeme.Surname",
           "Grammeme.Geo"]

user = ""
password = ""
host = ""
dataset = "dataset"
corpus = "corpus"
dialect = "mysql+mysqlconnector"
port = 3306
# for sql alchemy
params1 = "{}://{}:{}@{}:{}/{}".format(dialect,
                                      user,
                                      password,
                                      host,
                                      port,
                                      dataset)
params2 = "{}://{}:{}@{}:{}/{}".format(dialect,
                                      user,
                                      password,
                                      host,
                                      port,
                                      corpus)

PATH = "~/preprocessing"
MAX_FEATURES = 3000


def save(texts, output_name, labels=None, columns=None):
    path = output_name + ".pkl"
    s = "Saving to {}...".format(path)
    print(s)
    df = pd.DataFrame(columns=columns, data=texts)
    if labels is not None:
        df.insert(len(df.columns), "labels", labels)
    df.to_pickle(path)


def save_vectors(vectors, output_name):
    print("Saving vectors to ", output_name)
    with open(output_name + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(vectors))


def load_vectors(filename):
    with open(filename + ".json", "r") as f:
        vocab = json.loads(f.read())
    return vocab
