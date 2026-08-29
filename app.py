import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## streamlit app
st.title("Customer Churn Prediction")

## user input
# age = st.number_input("Age", min_value=0, max_value=120, value=30)
# gender = st.selectbox("Gender", options=["Male", "Female"])
# geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
# balance = st.number_input("Balance", min_value=0.0, value=0.0)
# credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
# tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
# has_cr_card = st.checkbox("Has Credit Card")
# is_active_member = st.checkbox("Is Active Member")

# user input

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

## prepare the input data for prediction only for multiple samples
# input_data = pd.DataFrame( {
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'Tenure': [tenure],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member]
#     'EstimatedSalary': [estimated_salary],
# } )

## one-hot encode the geography column
# geography_encoded = onehot_encoder_geo.transform([[geography]])
#geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Concatenate the input data with the one-hot encoded geography
# input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)



# Preprocess the input data
# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode geography
geography_encoded = onehot_encoder_geo.transform([[geography]])[0]

# Scale the input data
input_data = np.array([[gender_encoded, age, balance, credit_score, estimated_salary, tenure, num_of_products, has_cr_card, is_active_member,*geography_encoded]])
input_data = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data)

# Display the prediction result
if st.button("Predict"):
    if prediction[0][0] > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is unlikely to churn.")
