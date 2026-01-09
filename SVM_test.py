import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION (GE 1.6-82.5 TURBINE)
# ==========================================
TURBINE_CONFIG = {
    'name': 'GE 1.6-82.5 (Bac Lieu)',
    'rated_power': 1600,  # kW
    'cut_in': 3.5,        # m/s
    'rated_speed': 11.0,  # m/s
    'cut_out': 25.0,      # m/s
    'hub_height': 80,     # m
    'data_height': 100    # m
}

print(f"Initializing SVR model for: {TURBINE_CONFIG['name']}")

# ==========================================
# 2. PHYSICS FUNCTIONS
# ==========================================
def calculate_air_density(pressure_hpa, temp_c):
    R_specific = 287.058
    pressure_pa = pressure_hpa * 100
    temp_k = temp_c + 273.15
    return pressure_pa / (R_specific * temp_k)

def adjust_wind_height(v_data, h_data, h_hub):
    alpha = 0.11 
    return v_data * (h_hub / h_data) ** alpha

def get_turbine_power_baclieu(wind_speed, air_density):
    cfg = TURBINE_CONFIG
    rho_std = 1.225
    
    p_std = 0
    if wind_speed < cfg['cut_in'] or wind_speed > cfg['cut_out']:
        p_std = 0
    elif wind_speed >= cfg['rated_speed']:
        p_std = cfg['rated_power']
    else:
        ratio = (wind_speed - cfg['cut_in']) / (cfg['rated_speed'] - cfg['cut_in'])
        p_std = cfg['rated_power'] * (ratio ** 3)
    
    if p_std < cfg['rated_power'] - 5:
        p_corrected = p_std * (air_density / rho_std)
    else:
        p_corrected = cfg['rated_power']
        
    return p_std, p_corrected

# ==========================================
# 3. DATA PROCESSING & FEATURE ENGINEERING
# ==========================================
print("Loading and processing dataset...")
df = pd.read_csv('Dataset15years.csv')
df.columns = ['time', 'Temp', 'WindSpeed_kmh', 'WindDir', 'Pressure', 'Humidity']
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# A. Unit & Height Adjustment
v_100m = df['WindSpeed_kmh'] / 3.6
df['WindSpeed_Hub'] = adjust_wind_height(v_100m, TURBINE_CONFIG['data_height'], TURBINE_CONFIG['hub_height'])

# B. Feature Engineering (Lags for SVM)
# SVM needs flat features like XGBoost
target = 'WindSpeed_Hub'

# Cyclical Time features
df['Hour_Sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
df['Wx'] = np.cos(df['WindDir'] * np.pi / 180)
df['Wy'] = np.sin(df['WindDir'] * np.pi / 180)

# Create Lags (Simplified for SVM speed)
# SVM is slow, so we reduce the number of features compared to XGBoost
lags = [1, 2, 3, 6, 12, 24] 
for lag in lags:
    df[f'Lag_{lag}'] = df[target].shift(lag)

df_model = df.dropna().reset_index(drop=True)

# C. Scaling (CRITICAL FOR SVM)
# SVM works based on distance, so unscaled data will ruin it.
exclude_cols = ['time', 'WindSpeed_kmh', 'WindDir', target]
feature_cols = [c for c in df_model.columns if c not in exclude_cols]

print(f"Features used: {len(feature_cols)}")

X = df_model[feature_cols].values
y = df_model[target].values

# Split Train/Test
# WARNING: SVM training is O(n^2) or O(n^3). With 15 years (~130k rows), it will be very slow.
# For demonstration, I will train on the last 2 years (approx 17k rows).
# If you have a powerful PC, you can increase this limit.
train_size_limit = 20000 
if len(X) > train_size_limit:
    print(f"Dataset too large for SVM. Using last {train_size_limit} samples for training.")
    X = X[-train_size_limit:]
    y = y[-train_size_limit:]
    df_model = df_model.iloc[-train_size_limit:]

split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scaler Init
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ==========================================
# 4. SVM MODEL TRAINING
# ==========================================
print("\n>>> Training SVR Model (RBF Kernel)...")
print("This might take a minute depending on your CPU...")

# RBF Kernel is best for non-linear data like wind
model = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=0.1)

model.fit(X_train_scaled, y_train_scaled)

print("Training completed.")

# ==========================================
# 5. PREDICTION & PHYSICS CALCULATION
# ==========================================
print("\n>>> Calculating Real Power Output...")

# Predict (and inverse scale)
pred_scaled = model.predict(X_test_scaled)
pred_wind = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
actual_wind = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Get Environment Data
test_indices = df_model.index[split_idx:]
test_env_data = df_model.loc[test_indices, ['Temp', 'Pressure', 'time']].copy()

# Calculate Power
rho_values = []
power_std_list = []
power_corr_list = []

for w, temp, press in zip(pred_wind, test_env_data['Temp'], test_env_data['Pressure']):
    rho = calculate_air_density(press, temp)
    rho_values.append(rho)
    p_std, p_corr = get_turbine_power_baclieu(w, rho)
    power_std_list.append(p_std)
    power_corr_list.append(p_corr)

# Save results
test_env_data['Pred_Wind_Hub'] = pred_wind
test_env_data['Power_Standard'] = power_std_list
test_env_data['Power_Corrected'] = power_corr_list

# ==========================================
# 6. EVALUATION REPORT
# ==========================================
rmse = np.sqrt(mean_squared_error(actual_wind, pred_wind))
mae = mean_absolute_error(actual_wind, pred_wind)

total_std = sum(power_std_list)
total_corr = sum(power_corr_list)
diff_percent = ((total_corr - total_std) / total_std) * 100
avg_rho = np.mean(rho_values)

print("\n" + "="*50)
print(f"PERFORMANCE REPORT (Model: SVR - RBF)")
print("="*50)
print(f"1. Wind Speed RMSE:       {rmse:.4f} m/s")
print(f"2. Wind Speed MAE:        {mae:.4f} m/s")
print("-" * 50)
print(f"3. Avg Air Density:       {avg_rho:.3f} kg/m3")
print(f"4. Total Power (Real):    {total_corr/1000:.2f} MWh")
print(f"5. Revenue Deviation:     {diff_percent:.2f}%")
print("="*50)

# ==========================================
# 7. VISUALIZATION
# ==========================================
subset = test_env_data.iloc[:168]

fig, ax1 = plt.subplots(figsize=(15, 7))

# Wind Speed
ax1.set_xlabel('Time')
ax1.set_ylabel('Wind Speed @ 80m (m/s)', color='tab:blue')
ax1.plot(subset['time'], subset['Pred_Wind_Hub'], color='tab:blue', linestyle='--', alpha=0.7, label='Predicted (SVR)')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Power Output
ax2 = ax1.twinx()
ax2.set_ylabel('Power Output (kW)', color='tab:red')
ax2.plot(subset['time'], subset['Power_Corrected'], color='tab:red', linewidth=2, label='Power (Corrected)')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax2.fill_between(subset['time'], subset['Power_Standard'], subset['Power_Corrected'], 
                 color='yellow', alpha=0.3, label='Density Loss')

plt.title(f"Bac Lieu Wind Power Forecast (SVR Model)\nRMSE: {rmse:.2f} m/s | MAE: {mae:.2f} m/s")
ax2.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()