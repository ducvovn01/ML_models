import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

print(f"Initializing XGBoost model for: {TURBINE_CONFIG['name']}")

# ==========================================
# 2. PHYSICS FUNCTIONS
# ==========================================
def calculate_air_density(pressure_hpa, temp_c):
    """Calculate real-time air density (kg/m3)"""
    R_specific = 287.058
    pressure_pa = pressure_hpa * 100
    temp_k = temp_c + 273.15
    return pressure_pa / (R_specific * temp_k)

def adjust_wind_height(v_data, h_data, h_hub):
    """Adjust wind speed to hub height (Power Law)"""
    alpha = 0.11 
    return v_data * (h_hub / h_data) ** alpha

def get_turbine_power_baclieu(wind_speed, air_density):
    """Calculate Power Output (kW) with Air Density Correction"""
    cfg = TURBINE_CONFIG
    rho_std = 1.225
    
    # 1. Standard Power
    p_std = 0
    if wind_speed < cfg['cut_in'] or wind_speed > cfg['cut_out']:
        p_std = 0
    elif wind_speed >= cfg['rated_speed']:
        p_std = cfg['rated_power']
    else:
        # Cubic curve approximation
        ratio = (wind_speed - cfg['cut_in']) / (cfg['rated_speed'] - cfg['cut_in'])
        p_std = cfg['rated_power'] * (ratio ** 3)
    
    # 2. Correction
    if p_std < cfg['rated_power'] - 5:
        p_corrected = p_std * (air_density / rho_std)
    else:
        p_corrected = cfg['rated_power']
        
    return p_std, p_corrected

# ==========================================
# 3. DATA PROCESSING & FEATURE ENGINEERING
# ==========================================
print("Loading and processing 15-year dataset...")
df = pd.read_csv('Dataset15years.csv')
df.columns = ['time', 'Temp', 'WindSpeed_kmh', 'WindDir', 'Pressure', 'Humidity']
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# A. Unit & Height Adjustment
v_100m = df['WindSpeed_kmh'] / 3.6
df['WindSpeed_Hub'] = adjust_wind_height(v_100m, TURBINE_CONFIG['data_height'], TURBINE_CONFIG['hub_height'])

# B. Time Features (Cyclical)
df['Hour_Sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
df['Month_Sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
df['Wx'] = np.cos(df['WindDir'] * np.pi / 180)
df['Wy'] = np.sin(df['WindDir'] * np.pi / 180)

# C. Lag Features (Crucial for XGBoost)
# "What was the wind speed 1 hour ago? 24 hours ago?"
target = 'WindSpeed_Hub'
lags = [1, 2, 3, 6, 12, 24] # Lag steps
for lag in lags:
    df[f'Lag_{lag}'] = df[target].shift(lag)

# D. Rolling Features (Trends)
# "Average wind speed of the last 3 hours"
df['Roll_Mean_3'] = df[target].shift(1).rolling(window=3).mean()
df['Roll_Std_6'] = df[target].shift(1).rolling(window=6).std() # Volatility

# Drop NaNs created by shifting
df_model = df.dropna().reset_index(drop=True)

# Define Features and Target
exclude_cols = ['time', 'WindSpeed_kmh', 'WindDir', target]
feature_cols = [c for c in df_model.columns if c not in exclude_cols]

print(f"Input Features: {len(feature_cols)} features")
# print(feature_cols)

# Train/Test Split (90% - 10%)
split_idx = int(len(df_model) * 0.9)

X = df_model[feature_cols].values
y = df_model[target].values

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ==========================================
# 4. XGBOOST MODEL TRAINING
# ==========================================
print("\n>>> Training XGBoost Model...")

# 1. Khai báo early_stopping_rounds NGAY TẠI ĐÂY
model = XGBRegressor(
    n_estimators=1000,      
    learning_rate=0.01,     
    max_depth=6,            
    subsample=0.8,          
    colsample_bytree=0.8,   
    n_jobs=-1,              
    random_state=42,
    early_stopping_rounds=50  # <--- ĐƯA NÓ LÊN ĐÂY
)

# 2. Trong hàm fit() chỉ còn lại eval_set và verbose
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print(f"Best Iteration: {model.best_iteration}")
# ==========================================
# 5. PREDICTION & POWER CALCULATION
# ==========================================
print("\n>>> Calculating Real Power Output...")

# Predict Wind Speed
pred_wind = model.predict(X_test)
actual_wind = y_test

# Get Environment Data for Test Set
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
print(f"PERFORMANCE REPORT (Model: XGBoost)")
print("="*50)
print(f"1. Wind Speed RMSE:       {rmse:.4f} m/s")
print(f"2. Wind Speed MAE:        {mae:.4f} m/s")
print("-" * 50)
print(f"3. Avg Air Density:       {avg_rho:.3f} kg/m3 (Ref: 1.225)")
print(f"4. Total Power (Std):     {total_std/1000:.2f} MWh")
print(f"5. Total Power (Real):    {total_corr/1000:.2f} MWh")
print(f"6. Revenue Deviation:     {diff_percent:.2f}%")
print("="*50)

# ==========================================
# 7. VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Subplot 1: Forecast vs Actual (1 Week)
subset = test_env_data.iloc[:168]
ax1.set_xlabel('Time')
ax1.set_ylabel('Wind Speed (m/s)', color='tab:blue')
ax1.plot(subset['time'], subset['Pred_Wind_Hub'], color='tab:blue', linestyle='--', label='Predicted (XGBoost)')
ax1.axhline(y=TURBINE_CONFIG['rated_speed'], color='purple', linestyle=':', label='Rated Speed (11 m/s)')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')
ax1.set_title(f'XGBoost Forecast (RMSE: {rmse:.2f} m/s)')
ax1.grid(True, alpha=0.3)

# Subplot 2: Feature Importance (Unique to XGBoost)
# Get top 10 important features
importance = model.feature_importances_
feature_names = np.array(feature_cols)
sorted_idx = np.argsort(importance)[-10:]

ax2.barh(feature_names[sorted_idx], importance[sorted_idx], color='teal')
ax2.set_title('Top 10 Most Important Features')
ax2.set_xlabel('Relative Importance')

plt.tight_layout()
plt.show()