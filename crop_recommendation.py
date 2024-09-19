# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

# Train the model
@st.cache(allow_output_mutation=True)
def train_model():
    df = load_data()
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return rf_model, scaler, accuracy

# Main Streamlit app
def main():
    st.title("Crop Recommendation System")

    # Sidebar for user inputs
    st.sidebar.title("Enter Soil and Weather Parameters")
    
    N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
    P = st.sidebar.slider("Phosphorus (P)", 0, 145, 40)
    K = st.sidebar.slider("Potassium (K)", 0, 205, 40)
    temperature = st.sidebar.slider("Temperature (°C)", 10, 50, 20)
    humidity = st.sidebar.slider("Humidity (%)", 10, 100, 80)
    ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 10, 300, 200)

    # Display sample dataset and accuracy
    st.subheader("Sample Dataset")
    df = load_data()
    st.write(df.head())

    rf_model, scaler, accuracy = train_model()

    st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Predict crop based on user input
    if st.sidebar.button("Recommend Crop"):
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        input_data = scaler.transform(input_data)
        predicted_crop = rf_model.predict(input_data)
        st.write(f"Recommended Crop: **{predicted_crop[0]}**")

    # Feature importance plot
    if st.sidebar.checkbox("Show Feature Importance"):
        importances = rf_model.feature_importances_
        feature_names = df.drop('label', axis=1).columns
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, ax=ax)
        ax.set_title("Feature Importance in Crop Recommendation")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
