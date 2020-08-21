import streamlit as st
from methods import check_parameters
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (plot_confusion_matrix,
                             plot_roc_curve,
                             plot_precision_recall_curve)


class UserModel:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.LABELS_LIST = ["Negative", "Positive"]

    def fit_user_model(self):
        return self.model.fit(self.X_train, self.y_train)

    def metrics(self):
        st.subheader("Accuracy")
        st.write("Accuracy on train data: ",
                 round(self.model.score(self.X_train, self.y_train) * 100, 4), "%")
        st.write("Accuracy on validation data: ",
                 round(self.model.score(self.X_test, self.y_test) * 100, 4), "%")

        st.subheader("Confusion matrix (validation data)")
        plot_confusion_matrix(
            self.model, self.X_test,
            self.y_test, display_labels=self.LABELS_LIST
        )
        st.pyplot()

        st.subheader("ROC curve")
        plot_roc_curve(self.model, self.X_test, self.y_test)
        st.pyplot()

        st.subheader("Precision-Recall curve")
        plot_precision_recall_curve(self.model, self.X_test, self.y_test)
        st.pyplot()

    def __call__(self):
        self.fit_user_model()
        self.metrics()


def logistic_regression_model():
    st.sidebar.markdown("[Link to official documentation]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)")
    penalty = st.sidebar.selectbox("Penalty", (
        "l2", "l1", "elasticnet", 'none'
    ))
    dual = st.sidebar.selectbox("Dual", (
        False, True
    ))
    tol = st.sidebar.text_input("Tolerance", value="1e-4").replace(",", ".")
    C = st.sidebar.number_input("C (regularization)", min_value=0.,
                                max_value=100., value=1., step=.1)
    fit_intercept = st.sidebar.selectbox("Fit intercept", (
        True, False
    ))
    intercept_scaling = st.sidebar.number_input("Intercept scaling",
                                                min_value=0., max_value=100., value=1., step=.1)
    solver = st.sidebar.selectbox("Solver", (
        "lbfgs", "newton-cg", "liblinear",
        "sag", "saga"
    ))
    if check_parameters("Tolerance", tol, float):
        model = LogisticRegression(
            penalty=penalty, dual=dual, tol=float(tol),
            fit_intercept=fit_intercept, C=C,
            intercept_scaling=intercept_scaling, solver=solver
        )
        train_button = st.button("Train Logistic Regression model")
        return model, train_button
    else:
        return None, None


def support_vector_machine():
    st.sidebar.markdown("[Link to official documentation]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)")

    C = st.sidebar.number_input("C (regularization)", min_value=0.,
                                max_value=100., value=1., step=.1)
    kernel = st.sidebar.selectbox("Kernel", (
        "rbf", "linear", "poly",
        "sigmoid", "precomputed"
    ))
    degree = st.sidebar.slider("Degree of the poly function",
                               min_value=2, max_value=100, value=3, step=1)
    gamma = st.sidebar.radio("Kernel coefficient", (
        "scale", 'auto'
    ))
    coef0 = st.sidebar.number_input("Independent term in kernel function",
                                    min_value=0.0, max_value=1000000000.0, value=0.0, step=0.1)
    shrinking = st.sidebar.selectbox("Whether to use the shrinking heuristic", (
        True, False
    ))
    probability = st.sidebar.selectbox("Whether to enable probability estimates", (
        False, True
    ))
    tol = st.sidebar.text_input("Tolerance", value="1e-4")
    verbose = st.sidebar.selectbox("Enable verbose output", (
        False, True
    ))
    if check_parameters("Tolerance", tol, float):
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                    coef0=coef0, shrinking=shrinking, probability=probability,
                    tol=float(tol), verbose=verbose)
        train_button = st.button("Train SVM model")
        return model, train_button
    return None, None


def random_forest_classifier():
    st.sidebar.markdown("[Link to official documentation]\
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier)")
    n_estimators = st.sidebar.number_input("The number of trees in the forest",
                                           min_value=1, max_value=1000, value=100, step=1)
    criterion = st.sidebar.radio("Criteria", ("gini", "entropy"))
    max_depth = st.sidebar.number_input("The maximum depth of the tree (0 means None)",
                                        min_value=0, max_value=200, value=0, step=1)
    min_samples_split = st.sidebar.number_input("The minimum number of samples required to split",
                                                min_value=1, max_value=500, value=2, step=1)
    min_samples_leaf = st.sidebar.number_input("The minimum number of samples required to be at leaf node",
                                               min_value=1, max_value=500, value=1, step=1)
    min_weight_fraction_leaf = st.sidebar.number_input("The minimum weighted fraction of the sum total of weights",
                                                       min_value=0., max_value=.5, step=.1, value=0.)
    max_features = st.sidebar.radio("The number of features to best split", (
        "auto", "log2", "int", "float", None
    ))
    if max_features == "int":
        n_features = st.sidebar.number_input("Enter number of features",
                                             min_value=1, max_value=500, value=1, step=10)
    elif max_features == "float":
        n_features = st.sidebar.number_input("Enter number of features",
                                             min_value=1., max_value=500., value=1., step=10.)
    else:
        n_features = None

    max_leaf_nodes = st.sidebar.number_input("The maximum leaf nodes (0 means None)",
                                             min_value=0, max_value=200, value=0, step=1)
    min_impurity_decrease = st.sidebar.number_input("The minimum impurity decrease",
                                                    min_value=.0, max_value=10., value=0., step=.1)
    bootstrap = st.sidebar.radio("Use bootstrap samples for building trees", (True, False))
    oob_score = st.sidebar.radio("Use out-of-bag samples to estimate the generalization accuracy", (False, True))

    max_depth = None if max_depth == 0 else max_depth
    max_features = n_features if n_features is not None else max_features
    max_leaf_nodes = None if max_leaf_nodes == 0 else max_leaf_nodes
    if oob_score:
        bootstrap = True

    st.write("bootstrap params = ", bootstrap)
    st.write("Max depth param = ", max_depth)
    st.write("Max leaf nodes = ", max_leaf_nodes)
    train_button = st.button("Train Random Forest model", key='train_user_model')
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                   max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf,
                                   max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                   min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                                   oob_score=oob_score)
    return model, train_button
