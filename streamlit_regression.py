import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler

with open('label_encoder_gender_salary.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo_salary.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler_salary.pkl', 'rb') as f:
    scaler = pickle.load(f)

## streamlit app
st.title("Customer Estimated Salary Prediction")


# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
Exited = st.selectbox('Exited', [0, 1])

# Preprocess the input data
# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode geography
geography_encoded = onehot_encoder_geo.transform([[geography]])[0]

# Scale the input data
input_data = np.array([[credit_score, gender_encoded, age, tenure, balance, num_of_products, has_cr_card, is_active_member, Exited,*geography_encoded]])
input_data = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data)
#st.write("Predicted salary:", prediction[0][0])

# Display the prediction result
if st.button("Predict"):
    st.write("Estimated salary prediction:", prediction[0][0])