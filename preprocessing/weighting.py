"""Another feature weighing"""

from keras.preprocessing import sequence
from keras.preprocessing.text import \
    Tokenizer, one_hot
from sklearn.feature_extraction.text import \
    CountVectorizer, TfidfVectorizer
import logging
from gensim.models import Word2Vec, Doc2Vec
import subprocess
import numpy as np


def hashing(corpus, dim=1000):
    h = []
    for text in corpus:
        h.append(one_hot(text=text, n=dim))
    return sequence.pad_sequences(h)


def to_sequence(corpus, max_features=None):
    t = Tokenizer(num_words=max_features)
    t.fit_on_texts(corpus)
    seq = t.texts_to_sequences(corpus)
    return t.word_index, sequence.pad_sequences(seq)


def binary(corpus, max_features, min_df=1, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=min_df, max_features=max_features,
                                 ngram_range=ngram_range, binary=True)
    features = vectorizer.fit_transform(corpus)
    return vectorizer.get_feature_names(), features.toarray()


def frequency(corpus, max_features, min_df=1, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=min_df, max_features=max_features,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer.get_feature_names(), features.toarray()


def tf_idf(corpus, max_features, min_df=1, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    # document-term matrix
    return vectorizer.get_feature_names(), features.toarray()


# word2vec
def train_w2v(sentences, size, window, sample, min_ct, epochs):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    model = Word2Vec(sentences=sentences, size=size, window=window,
                     sample=sample, min_count=min_ct, workers=4)
    model.train(sentences=sentences, total_examples=len(sentences),
                epochs=epochs)
    print(model.wv.index2word)
    vectors = dict()
    for word in model.wv.index2word:
        vectors[word] = model.wv.get_vector(word).tolist()
    return vectors


# glove
def train_glove(corpus_name, size, window, min_ct, epochs):
    try:
        subprocess.call(["bash", "train_glove.sh", corpus_name,
                     str(size), str(window), str(min_ct), str(epochs)])
    except:
        print("Error!")
        return
    vectors = dict()
    with open("glove_vectors.txt", "r", encoding="utf-8") as f:
        for line in f:
            splitted = line.split()
            word = splitted[0]
            embedding = [float(val) for val in splitted[1:]]
            vectors[word] = embedding
    return vectors


# for embedding layer using glove и w2v
def build_embedding_matrix(index_vocab, vectors):
    size = len(list(vectors.items())[0][1])
    embedding_matrix = np.zeros((len(index_vocab) + 1, size))
    for word, index in index_vocab.items():
        vector = vectors.get(word)
        if vector is not None:
            embedding_matrix[index] = np.asarray(vector, dtype="float32")
    return embedding_matrix


def avg_wv(corpus, vectors):
    size = len(list(vectors.items())[0][1])
    averaged_vectors = np.zeros((corpus.shape[0], size), dtype="float32")
    nwords = 0
    for i in range(len(corpus)):
        for word in corpus[i].split():
            word_vec = vectors.get(word)
            if word_vec:
                averaged_vectors[i, :] = np.add(averaged_vectors[i], word_vec)
                nwords += 1
        if nwords:
            averaged_vectors[i, :] = np.divide(averaged_vectors[i], nwords)
    return averaged_vectors


# tf-idf & wv
def tfidf_avg_wv(corpus, vectors, max_feat):
    size = len(list(vectors.items())[0][1])
    features, matrix = tf_idf(corpus=corpus, max_features=max_feat)
    weighted_matrix = np.zeros((matrix.shape[0], size), dtype="float32")
    for i in range(matrix.shape[0]):
        matched_tfidf = 0
        for j in range(matrix.shape[1]):
            word = features[j]
            vector = vectors.get(word)
            if vector:
                vector = np.asarray(vectors.get(word), dtype="float32")
                weighted_matrix[i, :] = np.add(weighted_matrix[i], matrix[i, j] * vector)
                matched_tfidf += matrix[i, j]
        if matched_tfidf:
            weighted_matrix[i, :] = np.divide(weighted_matrix[i], matched_tfidf)
    return weighted_matrix
