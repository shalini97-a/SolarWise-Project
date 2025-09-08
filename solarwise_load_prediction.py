# SolarWise Load Prediction Model
# Internship Project: AICTE – Sustainable Energy and Efficiency

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Load Dataset
df = pd.read_csv('solar_load_sample.csv', parse_dates=['timestamp'])

# Step 2: Sort by Timestamp
df = df.sort_values('timestamp')

# Step 3: Feature Engineering - Lag Features
df['load_lag_1'] = df['load_kW'].shift(1)
df['solar_lag_1'] = df['solar_kW'].shift(1)

# Drop NaN rows
df = df.dropna()

# Step 4: Define Features and Target
X = df[['load_lag_1', 'solar_lag_1', 'solar_kW']]
y = df['load_kW']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Save Model and Scaler
joblib.dump(model, 'solarwise_load_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Preprocessing done and model trained successfully.")
