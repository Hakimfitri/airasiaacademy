import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("#Simple Advertising Prediction App")
st.write("This app predicts the **Advertising** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV= st.sidebar.slider('TV', 0.7, 	296.4, 	149.7)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 22.9)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 12.9)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Hakim.h5", "rb")) #rb: read binary

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
