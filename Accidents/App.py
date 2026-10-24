import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import matplotlib
from IPython import get_ipython
from PIL import Image

#load encoder and model
model = joblib.load('rf_model.joblib')
encoder = joblib.load('ordinal_encoder2.joblib')


# 1. serious injury 2. slight injury 3. fatal injury
st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚗", layout="wide")

options_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_types_collision = ['Vehicle with vehicle collision', 'Collision with roadside objects', 'Collision with pedestrians', 'Rollover', 'Collision with animals', 'Collision with roadside-parked vehicles', 'Unknown', 'Fall from vehicle', 'Other', 'Collision with train']
options_sex = ['Male', 'Female', 'Unknown']
options_education_level = ['Junior high school', 'Elementary School', 'High school', 'Unknown', 'Above High School', 'Writing & Reading', 'Illiterate']
options_services_years = ['Unknown', '2-5yrs', 'Above 10yrs', '5-10yrs', '1-2yrs', 'Below 1yr']
options_acc_area = ['Other', 'Office areas', 'Residential areas', 'Church areas', 'Industrial areas', 'School areas', 'Recreational areas', 'Outside rural areas', 'Hospital areas', 'Market areas', 'Rural village areas', 'Unknown', 'Office areas', 'Recreational areas']

features = ['Number_of_vehicles_involved', 'Number_of_casualties', 'Hour_of_Day', 'Type_of_collision', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Service_year_of_vehicle', 'Day_of_week', 'Area_accident_occured']

#title
st.markdown("ACCIDENT SEVERITY PREDICTION APP 🛣️", unsafe_allow_html=True)

def main():
    with st.form('road_traffic_severity_form'):
        st.subheader("Please enter the following outputs:")
        No_vehicles = st.slider("Number of vehicles involved:", 1,7, value = 0, format = "%d")
        No_casualties = st.slider("Number of casualties:", 1,8, value = 0, format = "%d")
        Hour = st.slider("Hour of the day:", 0,23, value = 0, format = "%d")
        collision = st.selectbox("Type of collision:", options=options_types_collision)
        Age_band = st.selectbox("Driver age group?", options=options_age)
        Sex = st.selectbox("Sex of the driver:", options=options_sex)
        Education = st.selectbox("Education of driver:", options=options_education_level)
        service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_years)
        Day_week = st.selectbox("Day of the week:", options=options_day)
        Accident_area = st.selectbox("Area of accident:", options=options_acc_area)
        submit = st.form_submit_button("Predict")

    if submit:
        input_array = np.array([collision, Age_band, Sex, Education, service_vehicle,
                                Day_week, Accident_area], ndmin=2)
        encoded_arr = list(encoder.transform(input_array).ravel())
        num_arr = [No_vehicles, No_casualties, Hour]
        pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)

        #predict the target from all the input features
        prediction = model.predict(pred_arr)

        if prediction == 0:
            st.write(f"The severity prediction is fatal injury💀")
        elif prediction == 1:
            st.write(f"The severity prediction is serious injury")
        else:
            st.write(f"The severity prediction is slight injury")

        st.write("Developed by Berto")

a,b,c = st.columns([0.2,0.6,0.2])
with b:
    st.image("bannerimage.jpg", width="stretch")

st.subheader("Description:")
st.text("""This data set is collected from Addis Ababa Sub-city police departments for master's
            research work.
            The data set has been prepared from manual records of road traffic accidents of the year 2017-20.
            All the sensitive information has been excluded during data encoding and finally it has 32 features 
            and 12316 instances of the accident.
            Then it is preprocessed and for identification of major causes of the accident by analyzing it using
            different machine learning classification algorithms.
            """)
st.subheader("Problem Statement:")
st.text(""" This target feature is Accident_severity which is a multi-class variable.
            The task is to classify this variable based on the other 31 features step-by-step by going through each day's task.
            The metric for evaluation will be f1-score.""")

if __name__ == '__main__':
    main()


