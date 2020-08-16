import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Constants
MODEL_PATH = "pretrained_models/"
POSITIVE_RESULT = "Maybe you have health problems😞. \
                I recommend that you undergo a medical examination"
NEGATIVE_RESULT = "You are not at risk of developing diabetes. Be healthy.😊"

# Load data for custom user models
@st.cache(persist=True)
def load_data():
    """ 
    Load data for training

    Returns:
        X (pd.DataFrame) - features, 
        y (pd.Series) - classes.
    """
    df = pd.read_csv("training_data/diabetes_data_upload.csv")
    df = encode_features(df)
    X = df.drop(columns=['class'])
    y = df['class']
    return X, y

def encode_result(num):
    if num == 1:
        return "Positive. " + POSITIVE_RESULT
    else:
        return "Negative. " + NEGATIVE_RESULT

def encode_features(df):
    """
    Encode categorical features in dataframe

    Parameters:
        df (pd.DataFrame): Pandas DataFrame with categorical features
    
    Returns:
        pd.DataFrame: Pandas DataFrame with encoded features
    """
    label = LabelEncoder()
    for col in df.columns:
        df[col] = label.fit_transform(df[col])
    return df

def make_predictions(model_name, data):
    """
    Load model and make predictions on user data

    Parameters:
        model_name (str): model filename without extension
        data (pd.DataFrame, array): data for prediction
    
    Returns:
        (int, array): class(es) that model predicts based on data
    """
    model_name += '.pickle'
    model = pickle.load(open(MODEL_PATH+model_name, 'rb'))
    return model.predict(data)

@st.cache(persist=True)
def split_data(X, y, test_size):
    """
    Split data into training and validation

    Parameters:
        X (pd.DataFrame): features
        y (pd.Series, array): targets

    Returns:
        list of train-test split inputs
    """
    return train_test_split(X, y, test_size=test_size)

    