from sklearn.metrics import accuracy_score
from learning import *
import os
import numpy as np
from copy import deepcopy
from learning.standard_classifiers import *

MODELS = "models/"
STANDARD = os.path.join(MODELS, "weights", "standard")


def load_model_(file):
    if file.endswith(".pkl"):
        classifier = joblib.load(file)
    else:
        classifier = load_model(file)
    return classifier


def select_objects(sample_size, train_size=0.8):
    ind = list(range(sample_size))
    np.random.shuffle(ind)
    num_train = np.round(sample_size * train_size).astype(int)
    return ind[0:num_train], ind[num_train:sample_size]


def averaged_voting(classifiers, X, y, need_round, *args):
    y_pred = []
    train_ind, test_ind = select_objects(len(X[0]))
    for i in range(len(classifiers)):
        clf = load_model_(classifiers[i])
        kwargs = args[i] if args else {}
        fit(X[i][train_ind], y[train_ind], clf, **kwargs)
        pred = predict(X[i][test_ind], clf,
                need_round, **kwargs)
        y_pred.append(pred)
    y_pred = np.mean(np.array(y_pred), axis=0)
    y_pred = np.round(y_pred).astype(int)
    return accuracy_score(y[test_ind], y_pred)


def weighted_voting(classifiers, X, y, weigths, need_round, *args):
    train_ind, test_ind = select_objects(len(X[0]))
    y_pred = np.zeros(len(test_ind))
    for i in range(len(classifiers)):
        kwargs = args[i] if args else {}
        clf = load_model_(classifiers[i])
        fit(X[i][train_ind], y[train_ind], clf, **kwargs)
        weighted_prediction = weigths[i] * \
            predict(X[i][test_ind], clf, need_round, **kwargs)
        y_pred += weighted_prediction
    y_pred = np.round(y_pred).astype(int)
    return accuracy_score(y[test_ind], y_pred)


def random_choice(size):
    ind = np.random.choice(size,
                           size=size,
                           replace=True)
    return ind


def bootstrap(n, sample, y):
    samples = []
    labels = []
    for i in range(n):
        ind = random_choice(len(sample))
        samples.append(sample[ind])
        labels.append(y[ind])
    return samples, labels


def bagging(classifiers, X, y, need_round, *args):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)
    samples, labels = \
        bootstrap(n=len(classifiers),
        sample=X_train, y=y_train)
    y_pred = np.zeros(y_test.shape[0])
    for i in range(len(samples)):
        clf = load_model_(classifiers[i])
        kwargs = args[i] if args else {}
        fit(samples[i], labels[i], clf, **kwargs)
        y_pred += predict(X_test, clf,
                          need_round,
                          **kwargs)
    y_pred = np.round(y_pred / len(samples)).astype(int)
    return accuracy_score(y_test, y_pred)


def transform_y(y):
    y_ = deepcopy(y)
    ind = np.where(y_ == 0)
    y_[ind] = -1
    return y_


def ada_boosting(classifiers, X, y, n, *args):
    X_test = []
    y_ = transform_y(y)
    train_ind, test_ind = select_objects(len(X[0]))
    D = np.full(len(train_ind), 1. / len(train_ind))
    composition = []
    args_ = ()
    while n:
        min_eps = 1.
        best_pred = []
        for i in range(len(classifiers)):
            clf = load_model_(classifiers[i])
            kwargs = args[i] if args else {}
            fit(X[i][train_ind], y[train_ind], clf, **kwargs)
            y_pred = predict(np.array(X[i])[train_ind], clf, True, **kwargs)
            epsilon = np.sum(D * (y[train_ind] != y_pred))
            if epsilon < min_eps:
                min_eps = epsilon
                best_pred = y_pred
                new_estimator = clf
                test_data = np.array(X[i])[test_ind]
                kwargs_ = kwargs
        theta = 0.5 * np.log((1 - min_eps) / min_eps)
        composition.append([theta, new_estimator])
        X_test.append(test_data)
        best_pred = transform_y(best_pred)
        args_ += (kwargs_,)
        D = D * np.exp(-theta * y_[train_ind] * best_pred)
        D /= np.sum(D)
        n -= 1
    acc = combine_predictions(X_test, y[test_ind], composition, *args_)
    return acc


def combine_predictions(X, y, params, *args):
    y_res = np.zeros(y.shape[0])
    i = 0
    for weight, clf in params:
        kwargs = args[i] if args else {}
        y_pred = predict(X[i], clf, need_round=True, **kwargs)
        y_res += weight * transform_y(y_pred)
        i += 1
    y_res = [0 if pred < 0 else 1 for pred in y_res]
    return accuracy_score(y, y_res)


def stacking(classifiers, X, y, decision_clf, *args):
    train_ind, test_ind = select_objects(len(X[0]))
    X_test = [np.array(X[i])[test_ind] for i in range(len(X))]
    predictions = np.zeros((len(train_ind), len(classifiers)))
    classifiers_ = []
    for i in range(len(classifiers)):
        clf = load_model_(classifiers[i])
        kwargs = args[i] if args else {}
        fit(np.array(X[i])[train_ind], y[train_ind], clf, **kwargs)
        classifiers_.append(clf)
        predictions[:, i] = predict(np.array(X[i])[train_ind], clf, False, **kwargs)
    fit(predictions, y[train_ind], decision_clf)
    acc = test_stacking_accuracy(classifiers_,
                                 X_test,
                                 y[test_ind],
                                 decision_clf,
                                 *args)
    return acc


def test_stacking_accuracy(classifiers, X, y, decision_clf, *args):
    predictions = np.zeros((len(y), len(classifiers)))
    for i in range(len(classifiers)):
        clf = classifiers[i]
        kwargs = args[i] if args else {}
        predictions[:, i] = predict(X[i], clf, False, **kwargs)
    y_pred = predict(predictions, decision_clf, True)
    return accuracy_score(y, y_pred)


# best classifiers
kwargs = {
        "batch_size": 100,
        "epochs": 14,
        "verbose": 2,
}


kwargs1 = {**{"lstm": True}, **kwargs}

clf1 = os.path.join(STANDARD, "linear_svm.pkl")
dataset1 = os.path.join(WEIGTHS, "generate_stems_tf_idf.pkl")
X1, y1, input_dim = load_dataset(dataset1)

clf2 = os.path.join(STANDARD, "random_forest.pkl")
dataset2 = os.path.join(WEIGTHS, "w2v_generate_tokens.pkl")
X2, y2, input_dim = load_dataset(dataset2)

clf3 = os.path.join(STANDARD, "extra_tree.pkl")
dataset3 = os.path.join(WEIGTHS, "w2v_tf_idf_generate_lems.pkl")
X3, y3, input_dim = load_dataset(dataset3)

clf4 = os.path.join(STANDARD, "logistic_regression.pkl")
X4, y4 = X3, y3

clf5 = os.path.join(STANDARD, "simple_nn.pkl")
X5, y5 = X3, y3

clf6 = os.path.join(STANDARD, "mlp.pkl")
X6, y6 = X3, y3

clf7 = os.path.join(MODELS, "weights/lstm/generate_lems_tf_idf.h5")
dataset4 = os.path.join(WEIGTHS, "generate_lems_tf_idf.pkl")
X7, y7, input_dim = load_dataset(dataset4)

clf8 = os.path.join(MODELS, "weights/stacked_lstm/w2v_tf_idf_generate_lems.h5")
X8, y8 = X3, y3

clf9 = os.path.join(MODELS, "weights/lstm/w2v_tf_idf_generate_lems.h5")
X9, y9 = X3, y3

clf10 = os.path.join(MODELS, "embedding/w2v/lstm/generate_stems.h5")
dataset5 = os.path.join(SEQUENCES, "generate_stems.pkl")
X10, y10, input_dim = load_dataset(dataset5)

clf11 = os.path.join(MODELS, "embedding/w2v/stacked_lstm/generate_stems.h5")
X11, y11 = X10, y10


comp1 = [clf2, clf3]
data1, labels1 = [X2, X3], y1

comp2 = [clf1, clf4]
data2, labels2 = [X1, X4], y1

comp3 = [clf5, clf6]
data3, labels3 = [X5, X6], y1

comp4 = [clf7, clf8, clf9]
data4, labels4 = [X7, X8, X9], y1

comp5 = [clf10, clf11]
data5, labels5 = [X10, X11], y1
"""
comp6 = comp1 + comp4
data6 = data1 + data4
res = np.zeros(5)
for i in range(5):
    acc = averaged_voting(comp6, data6, y1, False, {}, {}, kwargs1, kwargs1)
    print("Temp accuracy", acc)
    res[i] = acc
print(np.mean(res))"""
"""res = np.zeros(5)
for i in range(5):
    acc1 = averaged_voting(comp1, data1, labels1, False)
    print(acc1)
    acc2 = averaged_voting(comp2, data2, labels2, False)
    print(acc2)
    acc3 = averaged_voting(comp3, data3, labels3, False)
    print(acc3)
    acc4 = averaged_voting(comp4, data4, labels4, False, kwargs1, kwargs1)
    print(acc4)
    acc5 = averaged_voting(comp5, data5, labels5, False, kwargs, kwargs)
    print(acc5)
    res += np.array([acc1, acc2, acc3, acc4, acc5])
print(res/5)"""
"""dataset, labels = X2, y2
clf11 = os.path.join(MODELS, "weights/lstm/w2v_tf_idf_generate_lems.h5")
bagging_clf = [clf3, clf4, clf5, clf8, clf11]
total_score = 0
for i in range(5):
    acc = bagging(bagging_clf, dataset, labels, False, {}, {}, {}, kwargs1, kwargs1)
    print(acc)
    total_score += acc
print(total_score / 5)"""
"""
res = np.zeros(5)
for i in range(5):
    acc1 = ada_boosting(comp1, data1, labels1, 5)
    print(acc1)
    acc2 = ada_boosting(comp2, data2, labels1, 5)
    print(acc2)
    acc3 = ada_boosting(comp3, data3, labels1, 5)
    print(acc3)
    acc4 = ada_boosting(comp4, data4, labels1, 5, kwargs1, kwargs1, kwargs1)
    print(acc4)
    acc5 = ada_boosting(comp5, data5, labels1, 5, kwargs, kwargs)
    print(acc5)
    res += np.array([acc1, acc2, acc3, acc4, acc5])

print(res / 5)
"""
"""
# stacking
res = np.zeros(5)
for i in range(5):
    acc1 = stacking(comp1, data1, labels1, logistic_regression())
    print(acc1)
    acc2 = stacking(comp2, data2, labels1, logistic_regression())
    print(acc2)
    acc3 = stacking(comp3, data3, labels1, logistic_regression())
    print(acc3)
    acc4 = stacking(comp4, data4, labels1, logistic_regression(), kwargs1, kwargs1, kwargs1)
    print(acc4)
    acc5 = stacking(comp5, data5, labels1, logistic_regression(), kwargs, kwargs)
    print(acc5)
    res += np.array([acc1, acc2, acc3, acc4, acc5])
print(res / 5)
"""

acc = 0.
for i in range(5):
    acc += stacking(comp3+comp4, data3+data4, labels1, mlp_classifier(hid_layers=(3,), nb_epochs=300), {}, {}, kwargs1, kwargs1, kwargs1)
    print(acc)
print(acc/5)

