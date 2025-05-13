---

## 📌 Weather Forecasting Using LSTM and Streamlit

### 📝 Short Description

A deep learning-based web app that predicts the next 3 days’ temperatures for selected cities using real-time weather data and an LSTM neural network, with an interactive UI built in Streamlit.

---

## 📂 Project Structure

```bash
.
├── app.py                      # Streamlit web application
├── Dl.ipynb                    # Jupyter notebook for training the LSTM model
├── updated_weather_data.csv    # Dataset used for model reference
├── model.pth                   # Trained LSTM model
├── requirements.txt            # Required Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Features

* 🌐 Real-time weather data from OpenWeatherMap API
* 🤖 LSTM neural network for temperature forecasting
* 🧠 Training and architecture explained in Jupyter notebook
* 🎛️ Clean, interactive UI via Streamlit

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/weather-lstm-app.git
   cd weather-lstm-app
   ```

2. **Install the dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your OpenWeatherMap API key**
   Open `app.py` and replace the placeholder:

   ```python
   api_key = "YOUR_API_KEY"
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 🧠 Model

* Framework: **PyTorch**
* Architecture: **LSTM (Long Short-Term Memory)**
* Task: Predict next 3 days’ temperature based on current weather
* Training: Performed in `Dl.ipynb` using historical data

---

## 🌐 API

* OpenWeatherMap API is used to fetch real-time temperature data.
* Visit [https://openweathermap.org/api](https://openweathermap.org/api) to get your free API key.

---

## 📸 Demo (Optional)

Add a screenshot or GIF here if you'd like:

```
![App Screenshot](demo.png)
```

---

## 🧾 License

This project is open-source and licensed under the [MIT License](LICENSE).

---

