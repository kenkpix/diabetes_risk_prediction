import pickle
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "pretrained_models/"
POSITIVE_RESULT = " Maybe you have health problems😞. \
                I recommend that you undergo a medical examination"
NEGATIVE_RESULT = " You are not at risk of developing diabetes. Be healthy.😊"

def encode_result(num):
    if num == 1:
        return "Positive." + POSITIVE_RESULT
    else:
        return "Negative" + NEGATIVE_RESULT

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
