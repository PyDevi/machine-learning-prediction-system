import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")
preprocessing = joblib.load("preprocessing.joblib")

st.set_page_config(page_title= "House Price Prediction")

st.title("House Price Prediction")
st.write("Enter the house details below to predict the price")

with st.sidebar:
    st.write("This House price prediction window")
    st.write("Please provide the following details: ")

area = st.sidebar.number_input("Area(sq.ft)")

bedrooms = st.sidebar.number_input("Bedrooms", min_value= 1,max_value= 10, value= 3)

bathrooms = st.sidebar.number_input("Bathrooms", min_value= 1,max_value= 10, value= 3)

stories = st.sidebar.number_input("Stories", min_value= 1, max_value= 5, value= 2)

mainroad = st.sidebar.selectbox("Main Road", ["yes", "no"])

guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])

basement = st.sidebar.selectbox("Basement", ["yes", "no"])

hotwaterheating = st.sidebar.selectbox("Hot WaterHeating", ["yes", "no"])

airconditioning = st.sidebar.selectbox("Air Conditioning",["yes", "no"])

parking = st.sidebar.number_input("Parking", min_value= 0, max_value= 5, value= 1)

prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["unfurnished","semi-furnished", "furnished"])

if st.sidebar.button("Predict Price"):

    input_data = {
        "area" : area,
        "bedrooms" : bedrooms,
        "bathrooms" : bathrooms,
        "stories" : stories,
        "mainroad" : mainroad,
        "guestroom" : guestroom,
        "basement" : basement,
        "hotwaterheating" : hotwaterheating,
        "airconditioning" : airconditioning,
        "parking" : parking,
        "prefarea" : prefarea,
        "furnishingstatus" : furnishingstatus

    }

    input_df = pd.DataFrame([input_data])

    input_df[preprocessing["num_cols"]] = input_df[
            preprocessing["num_cols"]
            ].fillna(preprocessing["num_fill"])
     
    input_df[preprocessing["cat_cols"]] = input_df[
            preprocessing["cat_cols"]
            ].fillna(preprocessing["cat_fill"])
    
    for col, mapping in preprocessing["encoders"].items():
            
        input_df[col] = input_df[col].map(mapping)
     
    input_df = input_df[preprocessing["columns"]]
        
    prediction = model.predict(input_df)
        
    st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")

    