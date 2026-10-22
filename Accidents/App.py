import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import matplotllib
from IPython import get_ipython
from PIL import Image

#load encoder and model
model = joblib.load('rf_model.joblib')
encoder = joblib.load('ordinal_encoder2.joblib')

st.set_option('deprecation.showfileUploaderEncoding', False)

# 1. serious injury 2. slight injury 3. fatal injury
st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚗", layout="wide")

options_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_types_collision = ['Vehicle with vehicle collision', 'Collision with roadside objects', 'Collision with pedestrians', 'Rollover', 'Collision with animals', 'Collision with roadside-parked vehicles', 'Unknown', 'Fall from vehicle', 'Other', 'Collision with train']
options_sex = ['Male', 'Female', 'Unknown']
options_education_level = ['Junior High School', 'Elementary School', 'High school', 'Unknown', 'Above High School', 'Writing & Reading', 'Illiterate']
options_services_years = ['Unknown', '2-5yrs', 'Above 10yrs', '5-10yrs', 'Below 2yrs']
