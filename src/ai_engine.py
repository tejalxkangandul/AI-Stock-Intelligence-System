import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# CONFIG - Must match train_model.py exactly
FEATURES = ['Close', 'MA7', 'MA21', 'RSI', 'Returns', 'Volatility']
LOOKBACK = 60


def run_forecast(df):
    try:
        model_path = 'models/lstm_model.h5'
        if not os.path.exists(model_path):
            print("❌ AI Engine: Model file not found.")
            return None, None, 0.0

        model = load_model(model_path)

        # 1. Filter the 6 features
        data = df[FEATURES].values

        # 2. FRESH SCALER
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        if len(scaled_data) < LOOKBACK:
            print("❌ AI Engine: Not enough data for lookback.")
            return None, None, 0.0

        # 3. Prepare the last 60-day window
        last_window = scaled_data[-LOOKBACK:]
        current_batch = last_window.reshape(1, LOOKBACK, len(FEATURES))

        # 4. PREDICT
        prediction_scaled = model.predict(current_batch, verbose=0)

        # 5. UNSCALE
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = prediction_scaled[0, 0]
        unscaled_prediction = scaler.inverse_transform(dummy)[0, 0]

        # 6. CONFIDENCE CALCULATION
        # Calculate recent volatility vs historical average
        recent_volatility = df['Volatility'].iloc[-1]
        # We assume a base confidence of 95% for stable markets
        # High volatility subtracts from this score
        volatility_penalty = recent_volatility * 100
        confidence_score = max(50.0, min(99.2, 98.5 - volatility_penalty))

        print(
            f"✅ AI Engine: 1-Day Prediction: {unscaled_prediction} | Confidence: {confidence_score:.1f}%")

        # Return 3 values now: Prediction, Window, and Confidence
        return unscaled_prediction, last_window, confidence_score

    except Exception as e:
        print(f"❌ AI Engine Error: {e}")
        return None, None, 0.0
