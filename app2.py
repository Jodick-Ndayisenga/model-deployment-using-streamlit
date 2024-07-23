import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.datasets import load_iris

# Load the trained models
with open("rf_classifier.pkl", "rb") as rf_pickle:
    rf_classifier = pickle.load(rf_pickle)

with open("knn_classifier.pkl", "rb") as knn_pickle:
    knn_classifier = pickle.load(knn_pickle)

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

def prediction_rf(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = rf_classifier.predict(features)
    return iris.target_names[prediction][0]

def prediction_knn(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn_classifier.predict(features)
    return iris.target_names[prediction][0]

def main():
    st.title("Iris Flower Prediction")

    html_temp = """
    <div style="background-color: #FFFF00; padding: 16px">
    <h1 style="color: #000000; text-align: center;">Streamlit Iris Flower Classifier ML App</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    sepal_length = st.text_input("Sepal Length", "Type Here")
    sepal_width = st.text_input("Sepal Width", "Type Here")
    petal_length = st.text_input("Petal Length", "Type Here")
    petal_width = st.text_input("Petal Width", "Type Here")
    result_rf = ""
    result_knn = ""

    if st.button("Predict"):
        result_rf = prediction_rf(float(sepal_length), float(sepal_width), float(petal_length), float(petal_width))
        result_knn = prediction_knn(float(sepal_length), float(sepal_width), float(petal_length), float(petal_width))

    st.write('Random Forest Classifier Prediction: ', result_rf)
    st.write('K-Nearest Neighbors Classifier Prediction: ', result_knn)

if __name__ == '__main__':
    main()
