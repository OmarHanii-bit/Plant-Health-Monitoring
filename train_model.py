import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# File path
file_path = "plant_monitor_health_data.csv"

# =============================
# Step 1 - Load & Explore
# =============================
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
    exit()

# =============================
# Step 2 - Data Cleaning
# =============================
# Drop ID column and likely target-proxy 'Health_Score'
# We want to predict Status based on environmental factors
cols_to_drop = ['Plant_ID', 'Health_Score']
for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)
        print(f"Dropped column: {col}")

# Handle missing values
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode Categorical Variables (if any remain)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Mapping for {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split Features and Labels
target_col = "Health_Status"
if target_col not in df.columns:
    print(f"Error: Target column '{target_col}' not found.")
    exit()

X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"Target distribution:\n{y.value_counts()}")
print(f"Feature columns: {X.columns.tolist()}")

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save Preprocessors
preprocessors = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'columns': X.columns.tolist(),
    'cat_cols': cat_cols.tolist()
}
joblib.dump(preprocessors, "preprocessors.pkl")
print("Preprocessors saved as 'preprocessors.pkl'.")

# =============================
# Step 3 - Train Models
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n==== Random Forest Results ====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Neural Network
model_ffn = Sequential([
    Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_ffn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n==== Training Neural Network ====")
model_ffn.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0)
loss, accuracy = model_ffn.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

model_ffn.save("plant_health_model.h5")
print("Model saved as 'plant_health_model.h5'.")
