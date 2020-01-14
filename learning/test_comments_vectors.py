"""Learning from vector representation"""

from learning.build_models import *
import os
import glob
import itertools

if not os.path.exists("models/weights/"):
    os.makedirs("models/weights")
if not os.path.exists("models/sequences/"):
    os.makedirs("models/sequences/")
if not os.path.exists("models/embedding/"):
    os.makedirs("models/embedding/")


def build_standard_classifiers():
    MODELS_PATH = "models/weights/standard/"
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    build_nb(os.path.join(MODELS_PATH, "naive_bayes"))
    build_svm(kernel="linear", degree=1,
              model_path=os.path.join(MODELS_PATH, "linear_svm"))
    build_svm(kernel="rbf", degree=1,
              model_path=os.path.join(MODELS_PATH, "svm_rbf"))
    build_svm(kernel="poly", degree=7,
              model_path=os.path.join(MODELS_PATH, "svm_poly"))
    build_random_forest(n_estimators=100, max_depth=8,
                        model_path=os.path.join(MODELS_PATH, "random_forest"))
    build_extra_tree(n_estimators=100, max_depth=8,
                     model_path=os.path.join(MODELS_PATH, "extra_tree"))
    build_logistic_regression(os.path.join(MODELS_PATH, "logistic_regression"))
    build_knn(n_neighbors=20, model_path=os.path.join(MODELS_PATH, "knn"))
    layer = (500,)
    build_mlp(hid_layers=layer, activation="logistic", nb_epochs=2000,
              model_path=os.path.join(MODELS_PATH, "simple_nn"))
    layers = (1000, 500,)
    build_mlp(hid_layers=layers, activation="logistic", nb_epochs=2000,
              model_path=os.path.join(MODELS_PATH, "mlp"))


def test_standard_classifiers():
    MODEL_PATH = "models/weights/standard/"
    res = {}
    for file in glob.glob(MODEL_PATH + "*"):
        print(file)
        for dataset in glob.glob(WEIGTHS + "*"):
            print(dataset)
            X, y, input_dim = load_dataset(dataset)
            acc = cross_validation(X, y, file, n_splits=6)
            res[os.path.basename(file) + "_" +
                os.path.basename(dataset)] = acc
        print(res)

build_standard_classifiers()
test_standard_classifiers()


def build_nn_weights():
    LSTM_PATH = "models/weights/lstm/"
    if not os.path.exists(LSTM_PATH):
        os.makedirs(LSTM_PATH)
    STACKED_LSTM_PATH = "models/weights/stacked_lstm/"
    if not os.path.exists(STACKED_LSTM_PATH):
        os.makedirs(STACKED_LSTM_PATH)
    CNN_PATH = "models/weights/cnn/"
    if not os.path.exists(CNN_PATH):
        os.makedirs(CNN_PATH)
    for dataset in glob.glob(WEIGTHS + "*"):
        dataset_name = os.path.basename(dataset).split(".")[0]
        output_path = os.path.join(LSTM_PATH, dataset_name)
        build_lstm(dataset=dataset, model_path=output_path,
                   lstm_units=128, spatial_dropout=0.3)
        output_path = os.path.join(STACKED_LSTM_PATH,
                                   dataset_name)
        build_stacked_lstm(dataset=dataset,
                           model_path=output_path, lstm_units=32)
        output_path = os.path.join(CNN_PATH, dataset_name)
        build_cnn(dataset=dataset, model_path=output_path,
                  nb_filters1=32, nb_filters2=64,
                  kernel_size=3, pool_size=3,
                  units=128, rate=0.3)


def test_nn_weights():
    WEIGTHS_PATH = "models/weights/"
    kwargs = {"batch_size": 100,
              "epochs": 10,
              "verbose": 2}
    kwargs1 = {**{"lstm": 1}, **kwargs}
    kwargs2 = {**{"cnn": 1}, **kwargs}
    res = {}
    for (dirpath, dirnames, filenames) in os.walk(WEIGTHS_PATH):
        # [lstm, stacked_lstm, cnn]
        for dirname in dirnames:
            if "lstm" in dirname:
                args = kwargs1
            elif "cnn" in dirname:
                args = kwargs2
            else:
                continue
            for file in glob.glob(os.path.join(WEIGTHS_PATH, dirname) + "/*"):
                filename = os.path.basename(file.split(".")[0] + ".pkl")
                X, y, input_dim = load_dataset(os.path.join(WEIGTHS, filename))
                acc = cross_validation(X, y, file, n_splits=2, **args)
                res[dirname + "_" + filename] = acc
                print(res)

build_nn_weights()
test_nn_weights()

def build_seq():
    LSTM_PATH = "models/sequences/lstm/"
    if not os.path.exists(LSTM_PATH):
        os.makedirs(LSTM_PATH)
    STACKED_LSTM_PATH = "models/sequences/stacked_lstm/"
    if not os.path.exists(STACKED_LSTM_PATH):
        os.makedirs(STACKED_LSTM_PATH)
    CNN_PATH = "models/sequences/cnn/"
    if not os.path.exists(CNN_PATH):
        os.makedirs(CNN_PATH)
    for dataset in glob.glob(SEQUENCES + "*"):
        dataset_name = os.path.basename(dataset).split(".")[0]
        output_path = os.path.join(LSTM_PATH, dataset_name)
        build_lstm(dataset=dataset, model_path=output_path,
                   lstm_units=128, spatial_dropout=0.25, embedding=True,
                   embedding_dim=64)
        output_path = os.path.join(STACKED_LSTM_PATH, dataset_name)
        build_stacked_lstm(dataset=dataset, model_path=output_path,
                           lstm_units=32, embedding=True,
                           embedding_dim=64)
        output_path = os.path.join(CNN_PATH, dataset_name)
        build_cnn(dataset=dataset, model_path=output_path,
                  nb_filters1=32, nb_filters2=64,
                  kernel_size=3, pool_size=3,
                  units=128, rate=0.3, embedding=True,
                  embedding_dim=64)


def test_seq():
    SEQ_PATH = "models/sequences/"
    kwargs = {
        "epochs": 14,
        "batch_size": 100,
        "verbose": 2
    }
    res = {}
    for (dirpath, dirnames, files) in os.walk(SEQ_PATH):
        for dirname in dirnames:
            for file in glob.glob(os.path.join(SEQ_PATH, dirname) + "/*"):
                filename = os.path.basename(file.split(".")[0] + ".pkl")
                X, y, input_dim = load_dataset(os.path.join(SEQUENCES, filename))
                acc = cross_validation(X, y, file, n_splits=6, **kwargs)
                res[dirname + "_" + filename] = acc
            print(res)
build_seq()
test_seq()

def build_embedding():
    EMBEDDING_PATH = "models/embedding/"
    if not os.path.exists(EMBEDDING_PATH):
        os.makedirs(EMBEDDING_PATH)
    subdirs = ["glove", "w2v"]
    netw = ["lstm", "stacked_lstm", "cnn"]
    for path in itertools.product(subdirs, netw):
        dir = os.path.join(EMBEDDING_PATH, *path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    for (dirpath, dirnames, filenames) in os.walk(EMBEDDING):
        for dirname in dirnames:
            for file in glob.glob(os.path.join(EMBEDDING, dirname) + "/*"):
                filename = os.path.basename(file.split(".")[0])
                embedding_matrix = file
                sequence = os.path.join(SEQUENCES, os.path.basename(file))
                lstm_path = os.path.join(EMBEDDING_PATH, *(dirname, "lstm", filename))
                stacked_path = os.path.join(EMBEDDING_PATH, *(dirname, "stacked_lstm", filename))
                cnn_path = os.path.join(EMBEDDING_PATH, *(dirname, "cnn", filename))
                build_lstm(sequence, lstm_path, lstm_units=128,
                           spatial_dropout=0.3, embedding=True,
                           embedding_matrix=embedding_matrix)
                build_stacked_lstm(sequence, stacked_path, lstm_units=64,
                                   embedding=True, embedding_matrix=embedding_matrix)

                build_cnn(sequence, cnn_path, nb_filters1=64,
                          nb_filters2=128, kernel_size=3,
                          pool_size=3, units=100, rate=0.2,
                          embedding=True, embedding_matrix=embedding_matrix)


def test_embedding():
    EMBEDDING_PATH = "models/embedding/"
    kwargs = {
        "epochs": 16,
        "batch_size": 100,
        "verbose": 2
    }
    res = {}
    for (dirpath, dirnames, files) in os.walk(EMBEDDING_PATH):
        for dirname in dirnames:
            for (dirpath1, dirnames1, files1) in os.walk(os.path.join(EMBEDDING_PATH, dirname)):
                for dirname_ in dirnames1:
                    for file in glob.glob(os.path.join(EMBEDDING_PATH, *(dirname, dirname_)) + "/*"):
                        filename = os.path.basename(file.split(".")[0] + ".pkl")
                        X, y, input_dim = load_dataset(os.path.join(SEQUENCES, filename))
                        acc = cross_validation(X, y, file, n_splits=6, **kwargs)
                        res[dirname + dirname_ + "_" + filename] = acc
                    print(res)

build_embedding()
test_embedding()
