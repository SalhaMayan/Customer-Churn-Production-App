import streamlit as st
import joblib
import pandas as pd

# Load the trained Decision Tree model and scaler
model = joblib.load('DecisionTree_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set Streamlit page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom page style (background color and fonts)
st.markdown(
    """
    <style>
    body {
        background-color: #E8F5E9; /* Light mint green */
        color: #333333;
    }
    .stApp {
        background-color: #E8F5E9;
    }
    .stButton>button {
        background-color: #66BB6A; /* Calm green button */
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h2 style='text-align: center; color: #2E7D32;'>🌿 Customer Churn Prediction</h2>", unsafe_allow_html=True)
st.markdown("""
This application predicts customer churn using a Decision Tree Classifier, 
based on customer profile and behavioral attributes.
""")
st.markdown("---")
st.markdown("### Customer Profile")

# Create two columns
col1, col2 = st.columns(2)

# Input form
with st.form(key='customer_form'):
    with col1:
        Tenure = st.number_input('Tenure (Years)', min_value=0, max_value=20, value=2)
        CashbackAmount = st.number_input('Cashback Amount', min_value=0.0, value=0.5)
        WarehouseToHome = st.number_input('Warehouse to Home (Distance in km)', min_value=0, value=15)
        NumberOfAddress = st.number_input('Number of Addresses', min_value=0, value=2)
        Complain = st.selectbox('Has Complained?', [0, 1])

    with col2:
        DaySinceLastOrder = st.number_input('Days Since Last Order', min_value=0, value=85)
        OrderAmountHikeFromlastYear = st.number_input('Order Amount Hike From Last Year', min_value=0.0, value=0.3)
        SatisfactionScore = st.selectbox('Satisfaction Score (0 = Lowest, 5 = Highest)', [0, 1, 2, 3, 4, 5])
        NumberOfDeviceRegistered = st.number_input('Number of Devices Registered', min_value=0, value=1)
        CouponUsed = st.number_input('Number of Coupons Used', min_value=0, value=1)

    submit_button = st.form_submit_button(label='Predict')

# Prediction
if submit_button:
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

    customer_scaled = scaler.transform(customer_data)
    prediction = model.predict(customer_scaled)

    if prediction[0] == 1:
        st.error("❗ Prediction: The customer is likely to Churn!")
    else:
        st.success("✅ Prediction: The customer is likely to Stay.")
