import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------- UI --------------------
st.set_page_config(page_title="ANN Regression App", layout="centered")
st.title("ANN Regression Model - CGPA Prediction")

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    # change file name if needed
    data = pd.read_csv("student_lifestyle_100k.csv")
    return data

data = load_data()
st.subheader("Dataset Preview")
st.write(data.head())

# -------------------- Target Column --------------------
target_col = "CGPA"

# -------------------- Preprocessing --------------------
X = data.drop(columns=[target_col])
y = data[target_col]

# Handle categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- ANN Model --------------------
model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation="relu"))
model.add(Dense(1))  # Linear activation for regression

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

# -------------------- Train Button --------------------
if st.button("Train Model"):
    with st.spinner("Training ANN model..."):
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Prediction
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.success("Model Trained Successfully")
    st.write(f"Mean Squared Error: **{mse:.4f}**")
    st.write(f"Mean Absolute Error: **{mae:.4f}**")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("ANN Regression Model using TensorFlow & Streamlit")
