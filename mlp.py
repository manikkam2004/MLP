# ==============================
# IMPORTS
# ==============================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ==============================
# DATA GENERATION (6000 samples)
# ==============================
np.random.seed(42)

n = 6000
time = np.arange(n)

trend = 0.001 * time
seasonal = np.sin(0.02 * time)
noise = np.random.normal(0, 0.1, n)

series = trend + seasonal + noise

df = pd.DataFrame({"value": series})

# Inject missing values
df.loc[100:110, "value"] = np.nan
df["value"] = df["value"].fillna(method="ffill")

# ==============================
# NORMALIZATION
# ==============================
scaler = MinMaxScaler()
data = scaler.fit_transform(df[["value"]])

# ==============================
# CREATE SEQUENCES
# ==============================
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X, y = create_sequences(data, SEQ_LEN)

split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# ==============================
# SSM MODEL (SIMPLIFIED)
# ==============================
class SimpleSSM(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden)
        self.state = nn.Linear(hidden, hidden)
        self.gate = nn.GELU()
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.zeros(x.size(0), 64)
        for t in range(x.size(1)):
            inp = self.input_proj(x[:, t, :])
            h = self.gate(self.state(h) + inp)
        return self.out(h)

model = SimpleSSM()

# ==============================
# TRAINING
# ==============================
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(10):
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==============================
# SSM PREDICTION
# ==============================
model.eval()
with torch.no_grad():
    ssm_preds = model(X_test).numpy()

ssm_preds = scaler.inverse_transform(ssm_preds)
y_true = scaler.inverse_transform(y_test.numpy())

# ==============================
# BASELINE: XGBOOST
# ==============================
df["lag1"] = df["value"].shift(1)
df["lag3"] = df["value"].shift(3)
df["lag6"] = df["value"].shift(6)
df["lag12"] = df["value"].shift(12)

df = df.dropna()

features = ["lag1","lag3","lag6","lag12"]
Xb = df[features]
yb = df["value"]

split2 = int(0.8*len(Xb))
Xb_train, Xb_test = Xb[:split2], Xb[split2:]
yb_train, yb_test = yb[:split2], yb[split2:]

xgb = XGBRegressor(n_estimators=100)
xgb.fit(Xb_train, yb_train)

xgb_preds = xgb.predict(Xb_test)

# ==============================
# METRICS
# ==============================
def metrics(y, p):
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    return mae, rmse

ssm_mae, ssm_rmse = metrics(y_true, ssm_preds)
xgb_mae, xgb_rmse = metrics(yb_test.values, xgb_preds)

print("SSM MAE:", ssm_mae, "SSM RMSE:", ssm_rmse)
print("XGB MAE:", xgb_mae, "XGB RMSE:", xgb_rmse)
