import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("updated_weather_data.csv")

# Extract city names
cities = df['City'].unique()

# API Configuration
api_key = "358d4712052db8663c857eec875c24d6"
base_url = "http://api.openweathermap.org/data/2.5/weather"

# LSTM Model class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Load trained model
model = LSTMModel()
model.load_state_dict(torch.load(r"C:\Users\Krishna\Desktop\Weather_Prediction_Project\model.pth"))
model.eval()  # Set model to evaluation mode

# Prediction function
def predict_temperatures(city):
    # API Request
    response = requests.get(f"{base_url}?q={city}&appid={api_key}&units=metric")
    data = response.json()

    if response.status_code == 200:
        current_temp = data['main']['temp']

        # Prepare data for LSTM model
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_temp = scaler.fit_transform(np.array([current_temp]).reshape(-1, 1))
        temp_tensor = torch.tensor(scaled_temp, dtype=torch.float32)

        # Predict next 3 days' temperature
        predictions = []
        for _ in range(3):
            with torch.no_grad():
                predicted_temp = model(temp_tensor).item()
            temp_tensor = torch.cat((temp_tensor, torch.tensor([[predicted_temp]])))
            predictions.append(predicted_temp)

        # Inverse scaling
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return current_temp, predictions.flatten()
    else:
        return None, None

# Streamlit UI
st.title("Weather Prediction App")
selected_city = st.selectbox("Select a city to check the weather:", cities)

if st.button("Predict Temperature"):
    current_temp, predicted_temps = predict_temperatures(selected_city)
    if current_temp is not None:
        st.write(f"**Current Temperature in {selected_city}:** {current_temp}°C")
        st.write(f"**Predicted Temperatures for the next 3 days in {selected_city}:** {predicted_temps}°C")
    else:
        st.error(f"Error fetching data for {selected_city}.")
