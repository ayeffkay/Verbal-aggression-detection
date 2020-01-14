from preprocessing import PATH
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import load_model


PATH = PATH + "dataset_db/"
WEIGTHS = PATH + "weights/"
SEQUENCES = PATH + "sequences/"
EMBEDDING = PATH + "embedding/"
FREQ_FEATURES = PATH + "freq_features.pkl"


def cross_validation(X, y, file, n_splits=10, **kwargs):
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
    accuracy = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if file.endswith(".pkl"):
            classifier = joblib.load(file)
        else:
            classifier = load_model(file)
        fit(X_train, y_train, classifier, **kwargs)
        accuracy.append(evaluate(X_test, y_test, classifier, **kwargs))
    return np.mean(np.array(accuracy))


def simple_fit(X, y, file, **kwargs):
    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.2, random_state=42)
    if file.endswith(".pkl"):
        classifier = joblib.load(file)
    else:
        classifier = load_model(file)
    fit(X_train, y_train, classifier, **kwargs)
    return evaluate(X_test, y_test, classifier, **kwargs)


def fit(X, y, classifier, **kwargs):
    if kwargs:
        if kwargs.get("lstm"):
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        if kwargs.get("cnn"):
            X = np.expand_dims(X, axis=2)
        classifier.fit(X, y,
                       batch_size=kwargs.get("batch_size"),
                       epochs=kwargs.get("epochs"),
                       verbose=kwargs.get("verbose"))
    else:
        classifier.fit(X, y)




def evaluate(X, y, classifier, **kwargs):
    if hasattr(classifier, "score"):
        return classifier.score(X, y)
    if hasattr(classifier, "evaluate"):
        if kwargs.get("lstm"):
            X = np.reshape(X,
                (X.shape[0], 1, X.shape[1]))
        if kwargs.get("cnn"):
            X = np.expand_dims(X, axis=2)
        scores = classifier.evaluate(X, y,
                batch_size=kwargs.get("batch_size"))
        return scores[1]


def load_dataset(filename):
    data = pd.read_pickle(filename).get_values()
    X = np.array(data[:, :-1])
    y = np.array(data[:, -1]).astype(int)
    return X, y, X.shape[1]


def predict(X, classifier, need_round=False, **kwargs):
    if kwargs.get("lstm"):
        X = np.reshape(X,
                       (X.shape[0], 1, X.shape[1]))
    if kwargs.get("cnn"):
        X = np.expand_dims(X, axis=2)
    if not need_round and hasattr(classifier, "predict_proba"):
        predictions = classifier.predict_proba(X)[:, 1]
    else:
        predictions = np.array(classifier.predict(X))
        predictions = predictions.reshape(predictions.shape[0])
    if need_round:
        predictions = np.round(predictions).astype(int)
    return predictions
