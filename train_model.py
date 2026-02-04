from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 1. Configuration - Added Indian Blue-Chip Stocks
TICKERS = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS'  # Indian Stocks
]
FEATURES = ['Close', 'MA7', 'MA21', 'RSI', 'Returns', 'Volatility']
LOOKBACK = 60
PREDICTION_DAYS = 1


def prepare_data():
    all_X, all_y = [], []
    from src.data_engine import get_all_stock_data

    print("🚀 Gathering Deep Market Data for 1-Day Forecasts...")

    for t in TICKERS:
        df = get_all_stock_data(t)

        if df is None or df.empty:
            print(f"⚠️ Warning: No data found for {t}. Skipping...")
            continue

        print(f"✅ Processing unique patterns for {t}")

        # Extract features
        data = df[FEATURES].values

        # LOCAL SCALER: This is the "Gold" that prevents the same prediction for every stock
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        if len(scaled_data) <= LOOKBACK + PREDICTION_DAYS:
            continue

        # Create windows targeted at only 1 day ahead
        for i in range(LOOKBACK, len(scaled_data) - PREDICTION_DAYS):
            all_X.append(scaled_data[i-LOOKBACK:i, :])
            all_y.append(scaled_data[i, 0])

    X_final, y_final = np.array(all_X), np.array(all_y)

    if len(X_final) == 0:
        print("❌ CRITICAL ERROR: No training samples collected!")
        exit()

    return X_final, y_final

# --- EXECUTION ---


X_train, y_train = prepare_data()

# 1. THE "INDIVIDUALIST" BRAIN

model = Sequential([
    Input(shape=(LOOKBACK, len(FEATURES))),

    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dense(1)  # Predicts Tomorrow's Close Price
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=Huber(), metrics=['mae'])

early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

print(f"📊 Training on {len(X_train)} market samples...")


model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

# --- SAVE LOGIC ---
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

save_path = os.path.join(model_dir, 'lstm_model.h5')
model.save(save_path, save_format='h5')

print("-" * 30)
print(f"✅ SUCCESS: Model saved at: {os.path.abspath(save_path)}")
print("-" * 30)
