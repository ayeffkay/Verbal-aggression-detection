"""Classification by frequencies"""
from learning.build_models import *
import glob
from sklearn.externals import joblib
import os

MODELS_PATH = "models/features/"
X, y, input_dim = load_dataset(FREQ_FEATURES)


def build_models():
    if not os.path.exists("models/features/"):
        os.makedirs(MODELS_PATH)
    build_nb(os.path.join(MODELS_PATH, "naive_bayes"))
    build_svm(kernel="linear", degree=1,
              model_path=os.path.join(MODELS_PATH, "linear_svm"))
    build_svm(kernel="rbf", degree=1,
              model_path=os.path.join(MODELS_PATH, "svm_rbf"))
    build_svm(kernel="poly", degree=3,
              model_path=os.path.join(MODELS_PATH, "svm_poly"))
    build_random_forest(n_estimators=100, max_depth=8,
                        model_path=os.path.join(MODELS_PATH, "random_forest"))
    build_extra_tree(n_estimators=100, max_depth=8,
                     model_path=os.path.join(MODELS_PATH, "extra_tree"))
    build_logistic_regression(os.path.join(MODELS_PATH, "logistic_regression"))
    build_knn(n_neighbors=10, model_path=os.path.join(MODELS_PATH, "knn"))
    layer = (10,)
    build_mlp(hid_layers=layer, activation="logistic", nb_epochs=2000,
              model_path=os.path.join(MODELS_PATH, "simple_nn"))
    layers = (12, 5,)
    build_mlp(hid_layers=layers, activation="logistic", nb_epochs=2000,
              model_path=os.path.join(MODELS_PATH, "mlp"))
    build_lstm(dataset=FREQ_FEATURES,
               model_path=os.path.join(MODELS_PATH, "lstm"),
               lstm_units=32,
               spatial_dropout=0.4)
    build_stacked_lstm(dataset=FREQ_FEATURES,
                       model_path=os.path.join(MODELS_PATH, "stacked_lstm"),
                       lstm_units=16)
    build_cnn(dataset=FREQ_FEATURES,
              model_path=os.path.join(MODELS_PATH, "cnn"),
              nb_filters1=16, nb_filters2=64, kernel_size=3,
              pool_size=3, units=20, rate=0.3)


def test_standard_clf():
    res = {}
    for file in glob.glob(MODELS_PATH + "/*.pkl"):
        acc = cross_validation(X, y, file, n_splits=6)
        res[os.path.basename(file)] = acc
    print(res)


build_models()
test_standard_clf()


def test_deep_nn():
    res = {}
    kwargs = {"batch_size": 100,
              "epochs": 20,
              "verbose": 2}
    kwargs1 = {**kwargs, **{"lstm": True}}
    kwargs2 = {**kwargs, **{"cnn": True}}
    for file in glob.glob(MODELS_PATH + "/*.h5"):
        if "lstm" in file:
            acc = cross_validation(X, y, file, n_splits=6, **kwargs1)
        else:
            acc = cross_validation(X, y, file, n_splits=6, **kwargs2)
        res[os.path.basename(file)] = acc
    print(res)


# test_deep_nn()


def get_feat_importances():
    clf = "models/features/random_forest.pkl"
    clf = joblib.load(clf)
    clf.fit(X, y)
    print(clf.feature_importances_)


get_feat_importances()

