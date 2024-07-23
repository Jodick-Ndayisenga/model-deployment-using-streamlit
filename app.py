import pandas as pd  
import numpy as np  
import pickle  
import streamlit as st  
from sklearn.datasets import load_iris
from PIL import Image

# Load the trained models
with open("rf_classifier.pkl", "rb") as rf_pickle:
    rf_classifier = pickle.load(rf_pickle)

with open("knn_classifier.pkl", "rb") as knn_pickle:
    knn_classifier = pickle.load(knn_pickle)

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

st.title("Iris Flower Species Prediction")

st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# Predictions
rf_prediction = rf_classifier.predict(df)
knn_prediction = knn_classifier.predict(df)

# Display predictions
st.subheader('Random Forest Classifier Prediction')
st.write(iris.target_names[rf_prediction][0])

st.subheader('K-Nearest Neighbors Classifier Prediction')
st.write(iris.target_names[knn_prediction][0])
