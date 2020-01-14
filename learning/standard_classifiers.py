from sklearn.ensemble import RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def naive_bayes():
    print("Running NB...")
    nb = GaussianNB(priors=None)
    return nb


def support_vector_machines(kernel="linear", degree=3):
    print("Running SVM...")
    svm = SVC(C=1.0,
             kernel=kernel,
             degree=degree,
             gamma="auto",
             coef0=0.0,
             shrinking=True,
             probability=True,
             tol=0.001,
             cache_size=200,
             class_weight=None,
             verbose=False,
             max_iter=5000,
             decision_function_shape="ovr",
             random_state=None)
    return svm


def random_forest(n_estimators=100, max_depth=7):
    print("Running Random Forest...")
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                criterion='entropy',
                                max_depth=max_depth,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features='auto',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=4,
                                random_state=None,
                                verbose=0,
                                warm_start=False,
                                class_weight=None)
    return rf


def extra_trees(n_estimators=100, max_depth=7):
    print("Running Extra Trees...")
    trees = ExtraTreesClassifier(n_estimators=n_estimators,
                                criterion='gini',
                                max_depth=max_depth,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features='auto',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                bootstrap=False,
                                oob_score=False,
                                n_jobs=4,
                                random_state=None,
                                verbose=0,
                                warm_start=False,
                                class_weight=None)
    return trees


def logistic_regression():
    print("Running Logistic Regression...")
    regr = LogisticRegression(penalty='l2', dual=False,
                              tol=0.0001, C=1.0,
                              fit_intercept=True,
                              intercept_scaling=1,
                              class_weight=None,
                              random_state=None,
                              solver='liblinear',
                              max_iter=100,
                              multi_class='ovr',
                              verbose=0,
                              warm_start=False,
                              n_jobs=1)
    return regr


def knn(n_neighbors):
    print("Running K Nearest Neighbors...")
    neighbors = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights='uniform',
                               algorithm='auto',
                               leaf_size=30, p=2,
                               metric='minkowski',
                               metric_params=None,
                               n_jobs=4)
    return neighbors


def mlp_classifier(hid_layers, activation="logistic", nb_epochs=100):
    print("Running Multilayer Perceptron...")
    mlp = MLPClassifier(hidden_layer_sizes=hid_layers,
                        activation=activation,
                        solver='adam',
                        alpha=0.001,
                        batch_size=100,
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        power_t=0.5,
                        max_iter=nb_epochs,
                        shuffle=True,
                        random_state=None,
                        tol=0.0001,
                        verbose=False,
                        warm_start=False,
                        momentum=0.9,
                        nesterovs_momentum=True,
                        early_stopping=False,
                        validation_fraction=0.1,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08)
    return mlp


def decision_tree():
    dt = DecisionTreeClassifier(criterion='gini',
                                splitter='best',
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features=None,
                                random_state=None,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                class_weight=None,
                                presort=False)
    return dt
