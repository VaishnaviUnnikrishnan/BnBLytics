# train_model.py

import pickle
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# 1. Load data
df = pd.read_csv('AB_NYC_2019.csv')

# 2. Data Preprocessing
df = df.dropna(subset=['price'])  # Drop if price is missing
df = df[(df['price'] > 10) & (df['price'] < 1000)]  # Keep reasonable prices
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Encode categorical variable (only room_type is needed now)
encoder = LabelEncoder()
df['room_type'] = encoder.fit_transform(df['room_type'])

# 3. Define features and target
features = ['room_type', 'minimum_nights']   # ONLY two features now
X = df[features]
y = df['price']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train XGBoost model with improved parameters
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 7. Save model and encoder
with open('price_prediction_model.pkl', 'wb') as f:
    pickle.dump((model, encoder), f)

print("Model and encoder saved as 'price_prediction_model.pkl'")
