import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import plotly.express as px

# Load the model from the pickle file
with open("models/xgboost_classifier.pkl", 'rb') as file:
    model = pickle.load(file)

# Set up the app
st.title("Employee Burnout Prediction")
st.write("Enter the following details to predict employee burnout level")

# Create input fields with appropriate ranges and descriptions
user_input = []

# Designation (assuming levels 0-5, where 0 is entry level and 5 is executive)
designation = st.slider(
    "Designation Level",
    min_value=0,
    max_value=5,
    value=2,
    help="0: Entry Level, 1: Junior, 2: Mid-Level, 3: Senior, 4: Lead, 5: Executive"
)
user_input.append(designation)

# Resource Allocation (typical work hours per week, normalized)
resource_allocation = st.slider(
    "Resource Allocation",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Workload level (0.0: Very Low, 0.5: Moderate, 1.0: Very High)"
)
user_input.append(resource_allocation)

# Mental Fatigue Score
mental_fatigue = st.slider(
    "Mental Fatigue Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    help="Rate your mental fatigue (0: No fatigue, 10: Extreme fatigue)"
)
user_input.append(mental_fatigue)

# Company Type (0: Service, 1: Product, 2: Other)
company_type = st.selectbox(
    "Company Type",
    options=[0, 1, 2],
    format_func=lambda x: {0: "Service", 1: "Product", 2: "Other"}[x],
    help="Type of company you work for"
)
user_input.append(company_type)

# Work from Home Setup
wfh_setup = st.selectbox(
    "Work from Home Setup Available",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Do you have proper work from home setup?"
)
user_input.append(wfh_setup)

# Gender (0: Male, 1: Female, 2: Other)
gender = st.selectbox(
    "Gender",
    options=[0, 1, 2],
    format_func=lambda x: {0: "Male", 1: "Female", 2: "Other"}[x],
    help="Select your gender"
)
user_input.append(gender)

# Add a divider
st.divider()

# Predict the burnout level
if st.button('Predict Burnout Level'):
    prediction = model.predict([user_input])
    prediction_label = ['No Burnout', 'Risk of Burnout', 'Severe Burnout'][prediction[0]]
    
    # Display prediction with color-coded box
    color_map = {
        'No Burnout': 'green',
        'Risk of Burnout': 'orange',
        'Severe Burnout': 'red'
    }
    
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {color_map[prediction_label]}25;
        border: 2px solid {color_map[prediction_label]}'>
        <h3 style='color: {color_map[prediction_label]}; margin: 0;'>
        Predicted Burnout Level: {prediction_label}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Visualize the prediction
    fig = px.pie(
        values=[1],
        names=[prediction_label],
        title="Burnout Prediction",
        color_discrete_map={
            'No Burnout': 'green',
            'Risk of Burnout': 'orange',
            'Severe Burnout': 'red'
        }
    )
    st.plotly_chart(fig)

    # Add recommendations based on prediction
    st.subheader("Recommendations:")
    if prediction_label == 'No Burnout':
        st.write("✅ Keep maintaining your current work-life balance")
        st.write("✅ Continue with regular breaks and self-care practices")
        st.write("✅ Stay connected with your team and maintain open communication")
    elif prediction_label == 'Risk of Burnout':
        st.write("⚠️ Consider reducing your workload if possible")
        st.write("⚠️ Take regular breaks during work hours")
        st.write("⚠️ Discuss your concerns with your supervisor")
        st.write("⚠️ Practice stress-management techniques")
    else:  # Severe Burnout
        st.write("🚨 Urgent: Consider taking time off to recover")
        st.write("🚨 Seek professional help or counseling")
        st.write("🚨 Discuss workload reduction with your supervisor")
        st.write("🚨 Implement strict boundaries between work and personal life")


