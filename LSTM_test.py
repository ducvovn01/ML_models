import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
    'hub_height': 80,     # m (Hub height)
    'data_height': 100    # m (Data source height)
}

print(f"Initializing configuration for: {TURBINE_CONFIG['name']}")

# ==========================================
# 2. PHYSICS FUNCTIONS
# ==========================================

def calculate_air_density(pressure_hpa, temp_c):
    """
    Calculate real-time air density (kg/m3).
    """
    R_specific = 287.058
    pressure_pa = pressure_hpa * 100
    temp_k = temp_c + 273.15
    return pressure_pa / (R_specific * temp_k)

def adjust_wind_height(v_data, h_data, h_hub):
    """
    Adjust wind speed from data height to hub height using Power Law.
    Alpha = 0.11 (typical for offshore/coastal areas).
    """
    alpha = 0.11 
    return v_data * (h_hub / h_data) ** alpha

def get_turbine_power_baclieu(wind_speed, air_density):
    """
    Calculate Power Output (kW) considering Air Density Correction.
    """
    cfg = TURBINE_CONFIG
    rho_std = 1.225 # Standard Air Density (ISO)
    
    # 1. Calculate Standard Power (Theoretical)
    p_std = 0
    if wind_speed < cfg['cut_in'] or wind_speed > cfg['cut_out']:
        p_std = 0
    elif wind_speed >= cfg['rated_speed']:
        p_std = cfg['rated_power']
    else:
        # Cubic curve approximation for the ramp-up zone
        ratio = (wind_speed - cfg['cut_in']) / (cfg['rated_speed'] - cfg['cut_in'])
        p_std = cfg['rated_power'] * (ratio ** 3)
    
    # 2. Air Density Correction
    # If the turbine is not at full capacity, power is proportional to density.
    # If at rated power, the pitch system limits output, so no correction needed.
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

# Rename columns for clarity
df.columns = ['time', 'Temp', 'WindSpeed_kmh', 'WindDir', 'Pressure', 'Humidity']

df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# A. Unit Conversion & Height Adjustment
# km/h -> m/s
v_100m = df['WindSpeed_kmh'] / 3.6
# 100m -> 80m (Hub Height)
df['WindSpeed_Hub'] = adjust_wind_height(v_100m, TURBINE_CONFIG['data_height'], TURBINE_CONFIG['hub_height'])

# B. Feature Engineering (Cyclical Features)
# Wind Direction
wd_rad = df['WindDir'] * np.pi / 180
df['Wx'] = np.cos(wd_rad)
df['Wy'] = np.sin(wd_rad)

# Time (Hour & Month)
df['Hour_Sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
df['Month_Sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)

# Select Input Features
feature_cols = ['WindSpeed_Hub', 'Temp', 'Pressure', 'Humidity', 'Wx', 'Wy', 'Hour_Sin', 'Hour_Cos']
target_col = 'WindSpeed_Hub'

# Scaling
data = df[feature_cols].values
scaler_X = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler_X.fit_transform(data)

scaler_y = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_y.fit_transform(df[[target_col]].values)

# Create Sliding Window Dataset
def create_dataset(dataset, target, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        dataX.append(dataset[i:(i + time_step), :])
        dataY.append(target[i + time_step])
    return np.array(dataX), np.array(dataY)

time_step = 24 # Look back 24 hours
X, y = create_dataset(data_scaled, target_scaled, time_step)

# Train/Test Split (90% Train - 10% Test)
split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ==========================================
# 4. LSTM MODEL TRAINING
# ==========================================
print("\n>>> Training Bidirectional LSTM Model...")
model = Sequential()
model.add(Input(shape=(time_step, len(feature_cols))))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=40, batch_size=64, 
                    validation_split=0.1, callbacks=[early_stop], verbose=1)

# ==========================================
# 5. PREDICTION & POWER CALCULATION
# ==========================================
print("\n>>> Calculating Real Power Output...")

# 1. Predict Wind Speed
pred_scaled = model.predict(X_test)
pred_wind = scaler_y.inverse_transform(pred_scaled).flatten()
actual_wind = scaler_y.inverse_transform(y_test).flatten()

# 2. Get Corresponding Environmental Data for the Test Set
test_indices = df.index[split + time_step : split + time_step + len(pred_wind)]
test_env_data = df.loc[test_indices, ['Temp', 'Pressure', 'time']].copy()

# 3. Calculate Air Density & Power Output
rho_values = []
power_std_list = []
power_corr_list = []

for w, temp, press in zip(pred_wind, test_env_data['Temp'], test_env_data['Pressure']):
    # Calculate density
    rho = calculate_air_density(press, temp)
    rho_values.append(rho)
    
    # Calculate power
    p_std, p_corr = get_turbine_power_baclieu(w, rho)
    power_std_list.append(p_std)
    power_corr_list.append(p_corr)

# Store results
test_env_data['Pred_Wind_Hub'] = pred_wind
test_env_data['Air_Density'] = rho_values
test_env_data['Power_Standard'] = power_std_list
test_env_data['Power_Corrected'] = power_corr_list

# ==========================================
# 6. EVALUATION REPORT (ADDED MAE)
# ==========================================
# Calculate Metrics
rmse = np.sqrt(mean_squared_error(actual_wind, pred_wind))
mae = mean_absolute_error(actual_wind, pred_wind)  # <--- MAE ADDED HERE

total_std = sum(power_std_list)
total_corr = sum(power_corr_list)
diff_percent = ((total_corr - total_std) / total_std) * 100
avg_rho = np.mean(rho_values)

print("\n" + "="*50)
print(f"PERFORMANCE REPORT (Model: Bi-LSTM)")
print("="*50)
print(f"1. Wind Speed RMSE:       {rmse:.4f} m/s")
print(f"2. Wind Speed MAE:        {mae:.4f} m/s")  # <--- Printed Here
print("-" * 50)
print(f"3. Avg Air Density:       {avg_rho:.3f} kg/m3 (Ref: 1.225)")
print(f"4. Total Power (Std):     {total_std/1000:.2f} MWh")
print(f"5. Total Power (Real):    {total_corr/1000:.2f} MWh")
print(f"6. Revenue Deviation:     {diff_percent:.2f}%")
print("="*50)

# ==========================================
# 7. VISUALIZATION
# ==========================================
# Plot first 168 hours (1 week)
subset = test_env_data.iloc[:168]

fig, ax1 = plt.subplots(figsize=(15, 7))

# Left Axis: Wind Speed
ax1.set_xlabel('Time')
ax1.set_ylabel('Wind Speed @ 80m (m/s)', color='tab:blue')
ax1.plot(subset['time'], subset['Pred_Wind_Hub'], color='tab:blue', linestyle='--', alpha=0.7, label='Predicted Wind')
ax1.axhline(y=TURBINE_CONFIG['rated_speed'], color='purple', linestyle=':', label='Rated Speed (11 m/s)')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Right Axis: Power Output
ax2 = ax1.twinx()
ax2.set_ylabel('Power Output (kW)', color='tab:red')
ax2.plot(subset['time'], subset['Power_Standard'], color='gray', linestyle=':', label='Power (Theoretical)', alpha=0.5)
ax2.plot(subset['time'], subset['Power_Corrected'], color='tab:red', linewidth=2, label='Power (Corrected)')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Highlight the difference
ax2.fill_between(subset['time'], subset['Power_Standard'], subset['Power_Corrected'], 
                 color='yellow', alpha=0.3, label='Density Loss')

plt.title(f"Bac Lieu Wind Power Forecast (GE 1.6MW)\nRMSE: {rmse:.2f} m/s | MAE: {mae:.2f} m/s")
ax2.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()