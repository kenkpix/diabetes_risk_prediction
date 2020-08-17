import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from methods import (encode_result, encode_features, 
                    make_predictions, load_data)
from models import *

st.title("Early stage diabetes risk prediction")
st.subheader("This ML app provide your opportunity to check your \
             diabetes risk on early stage")

# Sidebar for picking or training models
st.sidebar.title("Machine Learning models")
st.sidebar.markdown("You can pick pre trained ML model here or train \
    her in real time. Also I provide tuning parameters online, \
    so you can build your own model.")
st.sidebar.markdown("My best model is Gradient Boosting Classifier, \
    with accuracy 97%. You can pick her or build your own and do \
    better then me😃.")

build_own_model = st.sidebar.checkbox("Build my own model")
X, y = load_data()

# Creating new model, tune parameters, and fit on data
if build_own_model:
    st.write("Choose model that will be trained. All parameters \
        set on default. Also, choose a size of validation data.")
    classifier = st.sidebar.selectbox("Classifier", (
        "Logistic Regression", "Support Vector Machine (SVM)"
    ))
    test_size = st.slider("Validation data size", min_value=0.01, 
        max_value=0.99, value=0.2, step=0.01)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
        )

    if classifier == "Logistic Regression":
        log_reg_model, train_button = logistic_regression_model()
        if train_button:
            st.write(f"{classifier} model choosed.")
            log_reg_model = UserModel(log_reg_model, X_train, 
                X_test, y_train, y_test)
            log_reg_model()

    elif classifier == "Support Vector Machine (SVM)":
        svm_model, train_button = support_vector_machine()
        if train_button:
            st.write(f"{classifier} model choosed.")
            svm_model = UserModel(svm_model, X_train, X_test, y_train, y_test)
            svm_model()
else:
    classifier = st.sidebar.selectbox("Classifier", (
    "Logistic Regression", "Support Vector Machine (SVM)",
    "Random Forest Classifier", "Gradient Boosting Classifier"
))

# Functions

# Using replace method instead of LabelEncoder 
# LabelEncoder can't encode features correctly
def make_df_for_prediction(values, cols):
    """
    Load user data into dataframe

    Parameters:
        values (list) - list of user inputs
        cols (list) - list of columns
    
    Returns:
        pd.DataFrame with user inputs
    """
    df = pd.DataFrame([values], columns=cols)
    df = df.replace(["Male", "Female", "Yes", "No"], [1, 0, 1, 0])
    return df

# If user want to use pre trained model
if not build_own_model:
    st.markdown("Answer the questions below to get your result")
# FEATURES
    age = st.number_input("What is your age?", min_value=1, max_value=100, step=1)

    gender = st.selectbox("What is your gender?", ("Male", "Female"), 
        key='gender')

    polyuria = st.selectbox("Do you have a polyuria condition?", (
        "Yes", "No"
    ), key='polyuria')
    st.markdown("Polyuria is excessive or an abnormally large production \
        or passage of urine (greater than 2.5 L or 3 L over 24 hours in adults).")

    polydipsia = st.selectbox("Do you have a polydipsia condition?", (
        "Yes", "No"
    ), key='polydipsia')
    st.markdown("Polydipsia is excessive thirst or excess drinking.")

    weight_loss = st.selectbox("Have you had sudden weight loss recently?", (
        "Yes", "No"
    ), key='weight_loss')

    weakness = st.selectbox("Did you feel weakness recently?", (
        "Yes", "No"
    ), key='weakness')

    polyphagia = st.selectbox("Do you have symptoms of polyphagia?", (
        "Yes", "No"
    ), key='polyphagia')
    st.markdown("Polyphagia is an abnormally strong sensation of hunger or \
        desire to eat often leading to or accompanied by overeating.")

    if gender == "Female":
        thrush = st.selectbox("Are you worried about thrush?", (
            "Yes", "No"
        ), key="thrush")
    else:
        thrush = 0

    vision = st.selectbox("Have you had your vision deteriorated?", (
        "Yes", "No"
    ), key='vision')

    itching = st.selectbox("Are you worried about itching?", (
        "Yes", "No"
    ), key='itching')

    irritating = st.selectbox("Are you irritable lately?", (
        "Yes", "No"
    ), key='irritating')

    delayed_healing = st.selectbox("Have you noticed that you take a \
        long time to heal during your illness?", 
        ("Yes", "No"), key='delayed_healing')

    paresis = st.selectbox("Do you feel muscle weakness?", (
        "Yes", "No"
    ), key='paresis')

    stiffness = st.selectbox("Do you feel muscle stiffness?", (
        "Yes", "No"
    ), key='stiffness')
    st.markdown("Muscle stiffness is when your muscles feel tight and you \
        find it more difficult to move than you usually do, especially after \
        rest. You may also have muscle pains, cramping, and discomfort.")

    balding = st.selectbox("Are you balding?", (
        "Yes", "No"
    ), key='balding')

    overweight = st.selectbox("Are you overweight?", (
        "Yes", "No"
    ), key='overweight')
# FEATURES END

    values = [age, gender, polyuria, polydipsia, 
         weight_loss, weakness, polyphagia, thrush, 
         vision, itching, irritating, delayed_healing, 
         paresis, stiffness, balding, overweight]

    df_for_prediction = make_df_for_prediction(values, X.columns)

# Generate predictions or training user models
    if st.button("Get Result"):
        st.write("Using pre trained model {}.".format(classifier))
        pred = make_predictions(classifier, df_for_prediction)
        st.write("Your result is {}".format(
            encode_result(pred)
        ))
        