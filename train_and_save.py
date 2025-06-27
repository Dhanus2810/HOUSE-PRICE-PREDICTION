# train_and_save.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/Bengaluru_House_Data.csv")

# Preprocess total_sqft
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df.dropna(subset=['total_sqft'], inplace=True)

# Extract BHK from 'size'
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)
df.dropna(subset=['bhk'], inplace=True)

# Create model DataFrame
df_model = df[['location', 'total_sqft', 'bath', 'bhk', 'price']].dropna()
df_model['location'] = df_model['location'].apply(lambda x: x.strip())
loc_stats = df_model['location'].value_counts()
df_model['location'] = df_model['location'].apply(lambda x: 'other' if loc_stats[x] <= 10 else x)

# One-hot encode categorical variable X, and assign y
df_dummies = pd.get_dummies(df_model.drop('price', axis=1))
X = df_dummies
y = df_model['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/house_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save feature names
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")