import streamlit as st
from sklearn.linear_model import LogisticRegression

def logistic_regression_model():
    st.sidebar.markdown("[Link to official documentation]\
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)")
    penalty = st.sidebar.selectbox("Penalty", (
        "l2 (default)", "l1", "elasticnet", 'none'
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
        "lbfgs (default)", "newton-cg", "liblinear",
        "sag", "saga"
    ))
    try:
        float(tol)
    except:
        st.sidebar.warning("Tolerance should be int or float number")