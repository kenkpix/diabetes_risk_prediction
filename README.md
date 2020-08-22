# Machine Learning App powered by [Streamlit](https://www.streamlit.io/)

## About ML app 

This app created to get a machine learning playground, work with models in scikit-learn, tuning parameters and watch how them affect on accuracy and other metrics.

Also, you can pick pre trained models, answer the questions and you will be get your result.

## Preview

Main page
![](https://github.com/kenkpix/diabetes_risk_prediction/blob/master/images/main_1.png)

Fit models with your parameters and get your results (accuracy, confusion matrix, etc.)
![](https://github.com/kenkpix/diabetes_risk_prediction/blob/master/images/training.png)

## Setup project

Firsly, setup your environement and install required packages:

```
pip install -r requirements.txt
```

Run streamlit app:

```
streamlit run app.py
```


### What needs to be fixed or added

* exception or help messages, when user try to choose incompatible parameters;
* new models from scikit-learn or other packages;
* saving user models;
* make charts more beautiful;
* manipulations with data;

### About data

The data set taken from UCI Machine Learning Repository, and include 512 samples. It's a very small data set, but i can't find a similar large set.

[Link to data](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)
