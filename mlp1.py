# ============================================================
# Advanced Time Series Forecasting with Neural State Space Models (SSMs)
# Complete Project Implementation in One File
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# -------------------------------
# 1. DATA GENERATION / LOADING
# -------------------------------
# Synthetic high-frequency dataset (sensor-like signal)
np.random.seed(42)
time = np.arange(0, 5000)
signal = np.sin(0.02 * time) + 0.5 * np.sin(0.05 * time) + np.random.normal(0, 0.1, len(time))
df = pd.DataFrame({"time": time, "value": signal})

# Normalize
df["value"] = (df["value"] - df["value"].mean()) / df["value"].std()

# -------------------------------
# 2. BASELINE MODEL (XGBoost)
# -------------------------------
def create_lag_features(series, lags=10):
    df_feat = pd.DataFrame({"y": series})
    for lag in range(1, lags+1):
        df_feat[f"lag_{lag}"] = df_feat["y"].shift(lag)
    return df_feat.dropna()

lags = 10
features = create_lag_features(df["value"], lags)
X = features.drop("y", axis=1).values
y = features["y"].values

tscv = TimeSeriesSplit(n_splits=5)
baseline_preds, baseline_true = [], []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    baseline_preds.extend(preds)
    baseline_true.extend(y_test)

baseline_mae = mean_absolute_error(baseline_true, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(baseline_true, baseline_preds))

# -------------------------------
# 3. NEURAL STATE SPACE MODEL (Simplified S4-like)
# -------------------------------
class SimpleSSM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super(SimpleSSM, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

# Prepare sequences
seq_len = 50
data = []
labels = []
series = df["value"].values
for i in range(len(series) - seq_len):
    data.append(series[i:i+seq_len])
    labels.append(series[i+seq_len])
data = np.array(data)
labels = np.array(labels)

train_size = int(0.8 * len(data))
X_train, X_test = data[:train_size], data[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Train SSM
model_ssm = SimpleSSM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_ssm.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model_ssm.train()
    optimizer.zero_grad()
    output = model_ssm(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate
model_ssm.eval()
with torch.no_grad():
    preds_ssm = model_ssm(X_test_t).squeeze().numpy()
    true_ssm = y_test_t.squeeze().numpy()

ssm_mae = mean_absolute_error(true_ssm, preds_ssm)
ssm_rmse = np.sqrt(mean_squared_error(true_ssm, preds_ssm))

# -------------------------------
# 4. RESULTS COMPARISON
# -------------------------------
print("\n=== Performance Comparison ===")
print(f"Baseline (XGBoost) - MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}")
print(f"SSM (Simplified S4) - MAE: {ssm_mae:.4f}, RMSE: {ssm_rmse:.4f}")
