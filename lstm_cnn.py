import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Activation, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow.keras.backend as K

# 1. SETUP & CONFIGURATION
# ==========================================
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

CONFIG = {
    'file_path': 'Dataset15years.csv',
    'sequence_length': 24,    # Input window (24 hours)
    'batch_size': 64,         # Increased for stability
    'epochs': 50,
    'learning_rate': 0.0005,
    'benchmark_rmse': 0.74,   # Your LSTM Target
    'benchmark_mae': 0.53     # Your LSTM Target
}

# Turbine Physics Config (GE 1.6-82.5)
TURBINE_CFG = {
    'rated_power': 1600,
    'cut_in': 3.5,
    'rated_speed': 11.0,
    'cut_out': 25.0,
    'hub_height': 80,
    'data_height': 100
}

print(">>> INITIALIZING HIGH-PERFORMANCE HYBRID MODEL TESTING...")

# 2. PHYSICS & HELPER FUNCTIONS
# ==========================================
def adjust_wind_height(v_data, h_data, h_hub):
    """Power Law for wind shear adjustment"""
    alpha = 0.11 # Surface roughness coefficient for open water/flat land
    return v_data * (h_hub / h_data) ** alpha

def calculate_air_density(pressure_hpa, temp_c):
    """Ideal Gas Law"""
    return (pressure_hpa * 100) / (287.058 * (temp_c + 273.15))

# 3. CUSTOM ATTENTION LAYER
# ==========================================
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1) # Attention weights over time axis
        output = x * a
        return K.sum(output, axis=1)

# 4. DATA PIPELINE (CRITICAL FOR LOW RMSE)
# ==========================================
print("--- Loading and Preprocessing Data ---")

try:
    df = pd.read_csv(CONFIG['file_path'])
    # Renaming columns to standard English names
    df.columns = ['time', 'Temp', 'WindSpeed_kmh', 'WindDir', 'Pressure', 'Humidity']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
except FileNotFoundError:
    print(f"ERROR: File {CONFIG['file_path']} not found. Please ensure the file is in the correct directory.")
    exit()

# 4.1 Physics-Informed Transformation
v_100m = df['WindSpeed_kmh'] / 3.6 # Convert to m/s
df['WindSpeed_Hub'] = adjust_wind_height(v_100m, TURBINE_CFG['data_height'], TURBINE_CFG['hub_height'])

# 4.2 Feature Engineering (The Secret Sauce)
target = 'WindSpeed_Hub'

# Lag Features: Give the model "memory" of the immediate past
df['Lag_1'] = df[target].shift(1)   # Previous hour
df['Lag_24'] = df[target].shift(24) # Previous day same hour

# Cyclic Time Features
df['Hour_Sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)

# Cyclic Wind Direction
df['Wx'] = np.cos(df['WindDir'] * np.pi / 180)
df['Wy'] = np.sin(df['WindDir'] * np.pi / 180)

# Drop NaNs created by lagging
df.dropna(inplace=True)

# Select features
features = ['WindSpeed_Hub', 'Temp', 'Pressure', 'Wx', 'Wy', 
            'Hour_Sin', 'Hour_Cos', 'Lag_1', 'Lag_24']

print(f"Features used: {features}")

# 4.3 Scaling
# X: Standard scaling (good for Neural Networks)
scaler_X = StandardScaler()
data_X = scaler_X.fit_transform(df[features].values)

# Y: MinMax scaling (0-1 range for output stability)
scaler_y = MinMaxScaler(feature_range=(0, 1))
data_y = scaler_y.fit_transform(df[[target]].values)

# 4.4 Sliding Window Dataset
def create_dataset(dataset_x, dataset_y, time_step=24):
    dataX, dataY = [], []
    for i in range(len(dataset_x) - time_step):
        dataX.append(dataset_x[i:(i + time_step), :])
        dataY.append(dataset_y[i + time_step])
    return np.array(dataX), np.array(dataY)

X, y = create_dataset(data_X, data_y, CONFIG['sequence_length'])

# 4.5 Train/Test Split (90% Train, 10% Test)
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training Shape: {X_train.shape}")
print(f"Testing Shape:  {X_test.shape}")

# 5. MODEL ARCHITECTURE (OPTIMIZED HYBRID)
# ==========================================
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Block 1: Lightweight CNN (Denoising)
    # Reduced filters and kernel size to avoid overfitting/oversmoothing
    x = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    # No MaxPooling to preserve temporal resolution!
    
    # Block 2: Deep Bi-LSTM (Temporal Learning)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    
    # Block 3: Attention Mechanism (Global Context)
    x = Attention()(x)
    
    # Block 4: Output
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Huber loss is more robust to outliers (wind gusts) than MSE
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), 
                  loss=tf.keras.losses.Huber(delta=1.0))
    return model

model = build_model((X_train.shape[1], X_train.shape[2]))

# 6. TRAINING
# ==========================================
print("\n--- Starting Training ---")
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 7. EVALUATION & VALIDATION
# ==========================================
print("\n--- Evaluating Model ---")
pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to get actual wind speeds (m/s)
pred_actual = scaler_y.inverse_transform(pred_scaled).flatten()
y_test_actual = scaler_y.inverse_transform(y_test).flatten()

# Calculate Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
mae = mean_absolute_error(y_test_actual, pred_actual)

print("\n" + "="*50)
print("FINAL PERFORMANCE REPORT")
print("="*50)
print(f"Hybrid Model RMSE: {rmse:.4f} m/s")
print(f"Benchmark RMSE:    {CONFIG['benchmark_rmse']} m/s")
print(f"Result:            {'PASSED (Better)' if rmse < CONFIG['benchmark_rmse'] else 'FAILED'}")
print("-" * 50)
print(f"Hybrid Model MAE:  {mae:.4f} m/s")
print(f"Benchmark MAE:     {CONFIG['benchmark_mae']} m/s")
print(f"Result:            {'PASSED (Better)' if mae < CONFIG['benchmark_mae'] else 'FAILED'}")
print("="*50)

# 8. VISUALIZATION
# ==========================================
# Extract date times for the test set
test_times = df['time'].iloc[split_idx + CONFIG['sequence_length'] : split_idx + CONFIG['sequence_length'] + len(pred_actual)]

plt.figure(figsize=(15, 6))
# Plot only first 150 hours for clarity
plt.plot(test_times[:150], y_test_actual[:150], color='black', label='Actual Wind Speed', linewidth=1.5)
plt.plot(test_times[:150], pred_actual[:150], color='#ff7f0e', label='Hybrid Model Forecast', linestyle='--', linewidth=2)

plt.title(f"Hybrid CNN-BiLSTM-Attention Forecast (RMSE: {rmse:.3f})", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Wind Speed (m/s)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Program finished successfully.")