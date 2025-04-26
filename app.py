import streamlit as st
import joblib
import pandas as pd

# Load the trained Decision Tree model and scaler
model = joblib.load(r'C:\Users\WINDOWS_2024\OneDrive\Salha2024\Desktop1\Advanced business data analysis_second semester_2025\Final Project\streamlit.app\DecisionTree_churn_model.pkl')
scaler = joblib.load(r'C:\Users\WINDOWS_2024\OneDrive\Salha2024\Desktop1\Advanced business data analysis_second semester_2025\Final Project\streamlit.app\scaler.pkl')

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Set page style
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: #333333;
    }
    .css-1d391kg {background-color: #ffffff;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h2 style='text-align: center;'>🔮 Customer Churn Prediction</h2>", unsafe_allow_html=True)
st.markdown("""
This application predicts customer churn using a Decision Tree Classifier, 
based on customer profile and behavioral attributes.
""")
st.markdown("---")
st.markdown("### Customer Profile")

# Create two columns
col1, col2 = st.columns(2)

with st.form(key='customer_form'):
    with col1:
        Tenure = st.number_input('Tenure (Years)', min_value=0, max_value=20, value=2)
        CashbackAmount = st.number_input('Cashback Amount', min_value=0.0, value=0.5)
        WarehouseToHome = st.number_input('Warehouse to Home (Days)', min_value=0, value=15)
        NumberOfAddress = st.number_input('Number of Addresses', min_value=0, value=2)
        Complain = st.selectbox('Has Complained?', [0, 1])

    with col2:
        DaySinceLastOrder = st.number_input('Days Since Last Order', min_value=0, value=85)
        OrderAmountHikeFromlastYear = st.number_input('Order Amount Hike From Last Year', min_value=0.0, value=0.3)
        SatisfactionScore = st.slider('Satisfaction Score', min_value=0, max_value=5, value=2)
        NumberOfDeviceRegistered = st.number_input('Number of Devices Registered', min_value=0, value=1)
        CouponUsed = st.selectbox('Coupon Used?', [0, 1])

    submit_button = st.form_submit_button(label='Predict')

# Prediction
if submit_button:
    # Create DataFrame for the new customer
    customer_data = pd.DataFrame([{
        'Tenure': Tenure,
        'CashbackAmount': CashbackAmount,
        'WarehouseToHome': WarehouseToHome,
        'NumberOfAddress': NumberOfAddress,
        'Complain': Complain,
        'DaySinceLastOrder': DaySinceLastOrder,
        'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
        'SatisfactionScore': SatisfactionScore,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'CouponUsed': CouponUsed
    }])

    # Scale the new customer data
    customer_scaled = scaler.transform(customer_data)

    # Predict churn
    prediction = model.predict(customer_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.error("❗ Prediction: The customer is likely to Churn!")
    else:
        st.success("✅ Prediction: The customer is likely to Stay.")
