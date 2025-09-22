import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


# -------------------------------
# Define custom class used in pipeline
# -------------------------------
class CuisineBinarizer(BaseEstimator, TransformerMixin):
    def _init_(self):
        self.mlb = MultiLabelBinarizer(sparse_output=True)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        X_split = X.apply(lambda x: [c.strip() for c in x.split(',')])
        self.mlb.fit(X_split)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        X_split = X.apply(lambda x: [c.strip() for c in x.split(',')])
        return self.mlb.transform(X_split)


# -------------------------------
# Load the pipeline and dataset
# -------------------------------
MODEL_PATH = 'Model/restaurant_rating_model (1).pkl'
DATA_PATH = 'Data/RestaurantData.csv'

try:
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Extract unique cities and cuisines
unique_cities = df['City'].dropna().unique().tolist()
cuisines_list = df['Cuisines'].dropna().str.split(',').explode().str.strip().unique().tolist()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍽️ Restaurant Rating Predictor")
st.markdown("*Plan your new restaurant and predict its potential rating*")

# Create main layout with slogans on the left and form on the right
slogan_col, form_col = st.columns([1, 2])

# Left column - Slogans
with slogan_col:
    st.markdown("### Why Use Our Predictor?")

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <div style="color: white; font-size: 16px; font-weight: bold; text-align: center;">
                🚀 Plan Before You Launch
            </div>
            <div style="color: #f0f0f0; font-size: 12px; text-align: center; margin-top: 5px;">
                Delight customers
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <div style="color: white; font-size: 16px; font-weight: bold; text-align: center;">
                💰 Smart Investment Decisions
            </div>
            <div style="color: #f0f0f0; font-size: 12px; text-align: center; margin-top: 5px;">
                Reduce business risks
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <div style="color: white; font-size: 16px; font-weight: bold; text-align: center;">
                📊 Data-Driven Planning
            </div>
            <div style="color: #f0f0f0; font-size: 12px; text-align: center; margin-top: 5px;">
                Make informed business decisions
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Right column - Input Form
with form_col:
    st.markdown("### Plan Your Restaurant")

    # Create a form for better organization
    with st.form("prediction_form"):
        # Location and Cuisine section
        st.markdown("#### 📍 Location & Cuisine")
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("City", ["Select a city"] + unique_cities, index=0)
        with col2:
            selected_cuisines = st.multiselect("Cuisines", cuisines_list, default=["Italian", "Chinese"])

        # Cost and Pricing section
        st.markdown("#### 💵 Pricing Strategy")
        col1, col2 = st.columns(2)
        with col1:
            avg_cost_inr = st.number_input("Average Cost for Two (₹)", min_value=0.0, value=1100.0, step=100.0)
        with col2:
            price_range = st.number_input("Price Range (1-4)", min_value=1, max_value=4, value=3, step=1)

        # Services section
        st.markdown("#### 🛎️ Services Offered")
        col1, col2, col3 = st.columns(3)
        with col1:
            has_table = st.selectbox("Table Booking", ["Yes", "No"])
        with col2:
            has_online = st.selectbox("Online Delivery", ["Yes", "No"])
        with col3:
            is_delivering = st.selectbox("Delivering Now", ["Yes", "No"])

        cuisines = ", ".join(selected_cuisines)

        # Submit button
        submitted = st.form_submit_button("Predict Rating", type="primary", use_container_width=True)

        if submitted:
            try:
                # Set default votes to 0 for new restaurants or use a reasonable default
                default_votes = 0  # New restaurant starts with 0 votes

                input_df = pd.DataFrame({
                    "City": [city if city != "Select a city" else "Other"],
                    "Cuisines": [cuisines if cuisines else "Unknown"],
                    "Average Cost for two": [avg_cost_inr],
                    "Has Table booking": [has_table],
                    "Has Online delivery": [has_online],
                    "Is delivering now": [is_delivering],
                    "Price range": [price_range],
                    "Votes": [default_votes]  # Set to 0 for new restaurants
                })

                # Convert Yes/No to string and fill missing if needed
                input_df['City'] = input_df['City'].fillna('Other').astype(str)
                input_df['Cuisines'] = input_df['Cuisines'].fillna('Unknown').astype(str)

                with st.spinner("🔄 Analyzing restaurant concept..."):
                    prediction = pipeline.predict(input_df)[0]

                # Display prediction with better styling
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                                padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                        <h2 style="color: #2E7D32; margin: 0;">⭐ {prediction:.2f}</h2>
                        <p style="color: #555; margin: 5px 0 0 0;">Predicted Rating</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )



            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")
                st.warning("⚠️ Please ensure all inputs are valid and match the expected format.")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 20px;">
        Built with ❤️ using Streamlit | Restaurant Planning & Rating Prediction System
    </div>
    """,
    unsafe_allow_html=True
)