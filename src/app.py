import streamlit as st
import pandas as pd
import pickle

# Load original dataset for dropdown options
df_original = pd.read_csv(r'C:\Users\hp\Documents\ml1\data\heart.csv')

# Load encoders, scaler, model
with open(r'C:\Users\hp\Documents\ml1\models\encoders.pickle', 'rb') as handle:
    label_encoders = pickle.load(handle)

with open(r'C:\Users\hp\Documents\ml1\models\scaler.pickle', 'rb') as handle:
    scaler = pickle.load(handle)

with open(r'C:\Users\hp\Documents\ml1\models\model.pickle', 'rb') as handle:
    model = pickle.load(handle)

st.title("Heart Disease Prediction App")

# User inputs
age = st.selectbox('Choose your age', sorted(df_original['Age'].unique()))
sex = st.selectbox('Select your gender', df_original['Sex'].unique())
ChestPainType = st.selectbox('Select Chest Pain Type', df_original['ChestPainType'].unique())
RestingBP = st.selectbox('Select your RestingBP', sorted(df_original['RestingBP'].unique()))
Cholesterol = st.selectbox('Select your Cholesterol', sorted(df_original['Cholesterol'].unique()))
FastingBS = st.selectbox('Select your FastingBS', df_original['FastingBS'].unique())
RestingECG = st.selectbox('Select your RestingECG', df_original['RestingECG'].unique())
MaxHR = st.selectbox('Select your MaxHR', sorted(df_original['MaxHR'].unique()))
ExerciseAngina = st.selectbox('Select your ExerciseAngina', df_original['ExerciseAngina'].unique())
Oldpeak = st.selectbox('Select your Oldpeak', sorted(df_original['Oldpeak'].unique()))
ST_Slope = st.selectbox('Select your ST_Slope', df_original['ST_Slope'].unique())

# Collect user inputs
user_input = {
    'Age': age,
    'Sex': sex,
    'ChestPainType': ChestPainType,
    'RestingBP': RestingBP,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'RestingECG': RestingECG,
    'MaxHR': MaxHR,
    'ExerciseAngina': ExerciseAngina,
    'Oldpeak': Oldpeak,
    'ST_Slope': ST_Slope
}

# Convert to DataFrame
df_input = pd.DataFrame([user_input])


for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    df_input[col] = label_encoders[col].transform(df_input[col])

# Scale numerical features
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df_input[num_cols] = scaler.transform(df_input[num_cols])

b = st.button('submit to see prediction')
if b:


    pred = model.predict(df_input)
    if pred==1:
        st.write('Likely to have de...')
    else:
        st.success('Do not have de..')