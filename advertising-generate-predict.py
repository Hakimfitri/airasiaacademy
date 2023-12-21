%%writefile advertising-generate-predict.py
import streamlit as st
import pandas as pd
import pickle

st.write("# Simple Advertising Prediction App")
st.write("This app predicts the **Advertising** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.7, 296.4, 149.7)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 22.9)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 12.9)
    
    # Add a placeholder value for the missing feature
    MissingFeature = st.sidebar.slider('MissingFeature', 0.0, 100.0, 50.0)

    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,
            'MissingFeature': MissingFeature
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

file_path = "Hakim.h5"

try:
    with open(file_path, "rb") as file:
        loaded_model = pickle.load(file)

    # Ensure df is a 2D array for prediction
    prediction = loaded_model.predict(df.values)
    
    st.subheader('Prediction')
    st.write(prediction)

except FileNotFoundError:
    st.error("Model file 'Hakim.h5' not found. Please make sure the file exists.")
except Exception as e:
    st.error(f"An error occurred: {e}")
