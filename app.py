import streamlit as st 
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Load the trained model
model=tf.keras.models.load_model('ANN CLASSIFICATION\model.h5')

#Load the encoders and scaler
with open('ANN CLASSIFICATION\onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)
    
with open('ANN CLASSIFICATION\label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
    
with open('ANN CLASSIFICATION\scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
##Streamlit app
st.title("Customer Churn Prediction")

#input
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.number_input('Age',min_value=18,max_value=100)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#Prepare the input data
input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Geography': [geography] 
})

#one-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
input_data['Geography'] = geo_encoded
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine one-hot encoded columns with input data
input_data=input_data.drop(columns=['Geography'])
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data
input_data=input_data[scaler.feature_names_in_]
input_data_scaler=scaler.transform(input_data)

#Prediction
prediction=model.predict(input_data_scaler)
prediction_prob=prediction[0][0]

if prediction_prob>0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')