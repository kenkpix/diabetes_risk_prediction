import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    
    def __call__(self):
        self.fit_user_model()
        self.metrics()

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

def logistic_regression_model():
    st.sidebar.markdown("[Link to official documentation]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)")
    penalty = st.sidebar.selectbox("Penalty", (
        "l2", "l1", "elasticnet", 'none'
    ))
    dual = st.sidebar.selectbox("Dual", (
        False, True
    ))
    tol = st.sidebar.text_input("Tolerance", value="1e-4")
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
    try:
        float(tol.replace(",", "."))
    except:
        st.sidebar.warning("Tolerance should be int or float number")
    else:
        model = LogisticRegression(
            penalty=penalty, dual=dual, tol=float(tol),
            fit_intercept=fit_intercept, C=C,
            intercept_scaling=intercept_scaling, solver=solver
        )
        train_button = st.button("Train my model", key='train_user_model')
        return model, train_button
    return st.error("Something wrong with your parameters"), None

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
    try:
        float(tol.replace(",", "."))
    except:
        st.sidebar.warning("Tolerance should be int or float number")
    else:
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, shrinking=shrinking, probability=probability,
            tol=float(tol), verbose=verbose)
        train_button = st.button("Train my model", key='train_user_model')
        return model, train_button
    return st.error("Something wrong with your parameters"), None