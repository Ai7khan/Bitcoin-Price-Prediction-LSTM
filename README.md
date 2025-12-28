# 📈 Bitcoin Price Predictor using LSTM (Deep Learning)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project leverages **Deep Learning** to forecast cryptocurrency price trends. By engineering a Recurrent Neural Network (RNN) using **Long Short-Term Memory (LSTM)** architecture, the model is capable of learning long-term dependencies in volatile time-series data.

The system automates the entire data pipeline—fetching live market data via the **Yahoo Finance API**, normalizing it for neural network processing, and generating rolling window predictions.

## 🚀 Key Features
* **Deep Learning Architecture:** Utilizes a multi-layered LSTM network with Dropout regularization to prevent overfitting and capture complex market patterns.
* **High Accuracy:** Achieved **95.6% forecasting accuracy** (MAPE: 4.43%) on unseen test data from 2024–2025.
* **Real-Time Data:** Integrates `yfinance` to pull up-to-the-minute Bitcoin (BTC-USD) market data.
* **Dynamic Visualization:** Generates professional financial plots to compare predicted trends against actual market movements.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn (MinMaxScaler)
* **Visualization:** Matplotlib
* **Data Source:** Yahoo Finance API

## 📊 Results

### 1. Prediction Performance
The model was rigorously tested on a 12-month period (Jan 2024 – Jan 2025) containing significant market volatility.

![Bitcoin Price Prediction](prediction_result.png)
* *Green Line: Predicted Price (LSTM)*
* *Black Line: Actual Market Price*

**Performance Metrics:**
* **Mean Absolute Percentage Error (MAPE):** `4.43%` (Indicates strong trend tracking)
* **Mean Absolute Error (MAE):** `$2,737.83`
* **Root Mean Squared Error (RMSE):** `$4,871.95`
*(Note: RMSE is higher than MAE due to the model's sensitivity to sudden, extreme volatility spikes common in crypto markets.)*

### 2. Training Stability
To ensure the model learned generalizable patterns, I monitored the Training vs. Validation Loss.

![Loss Plot](loss_plot.png)
* The convergence of the Training Loss (Blue) and Validation Loss (Orange) confirms the model **did not overfit**, maintaining stability even when introduced to new data.

## 🧠 Model Architecture
The Neural Network is designed to look back at the previous **60 days** of data to predict the next day's closing price:
1.  **LSTM Layer 1:** 50 units, Return Sequences=True (Captures sequential patterns).
2.  **Dropout:** 20% (Prevents overfitting).
3.  **LSTM Layer 2:** 50 units, Return Sequences=False (Consolidates features).
4.  **Dropout:** 20%.
5.  **Dense Layer:** Output layer (Predicts the final price value).

## 💻 How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ai7khan/Bitcoin-Price-Predictor.git](https://github.com/YOUR_USERNAME/Bitcoin-Price-Predictor.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas yfinance matplotlib scikit-learn tensorflow
    ```
3.  **Run the script:**
    ```bash
    python main.py
    ```

## 🔮 Future Improvements
* **Multi-Asset Support:** Expand the model to predict Ethereum (ETH) and Solana (SOL).
* **Sentiment Analysis:** Integrate Twitter/X API data to weight predictions based on social sentiment.
* **Web Deployment:** Deploy the model as an interactive web app using **Streamlit**.

---
*Created by [Aibek Serikkali](https://linkedin.com/in/aibek-serikkali) - Computer Engineering Student at Ankara University.*
