
import os
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import (Input, Dense, LSTM, Reshape, Concatenate, 
                                     Add, Subtract, Multiply, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

# --- 1. SYSTEM CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
np.random.seed(42)
tf.random.set_seed(42)

CONFIG = {
    'window_size': 24,
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.0003,
    
    # BAC LIEU WIND SPECS
    'rotor_diameter': 82.5,   # For Wake Calculation
    'max_wind_speed': 25.0,   # For Normalization (Cut-out speed)
    'nwp_noise_level': 0.4    # Higher noise to force AI to learn correction
}

print(f">>> Initializing Wind Speed Forecaster (Target: m/s)...")

# ==============================================================================
# SECTION 2: PHYSICS SIMULATION (WIND FIELD DYNAMICS)
# ==============================================================================

def generate_bac_lieu_layout():
    """Generates the coordinates for 72 turbines."""
    coords = []
    for r in range(7):
        for c in range(8): coords.append([c * 500, r * 500])
    off_x, off_y = 8*500 + 1000, 500
    for r in range(4):
        for c in range(4): coords.append([off_x + c * 500, off_y + r * 500])
    return np.array(coords)

turbine_coords = generate_bac_lieu_layout()
N_TURBINES = len(turbine_coords)

def compute_jensen_adjacency(wind_dir_deg, coords, D=82.5, k=0.075):
    """Calculates Jensen Wake Adjacency Matrix."""
    n = len(coords)
    adj = np.zeros((n, n))
    theta = np.radians(90 - wind_dir_deg)
    rot_mat = np.array([[np.cos(theta), np.sin(theta)], 
                        [-np.sin(theta), np.cos(theta)]])
    coords_rot = coords @ rot_mat.T
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            dx = coords_rot[j, 0] - coords_rot[i, 0]
            dy = coords_rot[j, 1] - coords_rot[i, 1]
            if dx > 0:
                r_wake = (D / 2) + k * dx
                if abs(dy) < r_wake:
                    adj[j, i] = (D / (D + 2 * k * dx))**2
    return adj

def simulate_ground_truth(base_ws, base_wd, coords):
    """
    Simulates the Effective Wind Speed.
    Returns: The MEAN wind speed across all turbines after wake losses.
    """
    # 1. Calculate Wake Interactions
    adj = compute_jensen_adjacency(base_wd, coords)
    total_deficit_sq = np.sum(adj**2, axis=1)
    
    # 2. Calculate Effective Speed at each turbine
    # V_eff = V_free * (1 - sqrt(sum(deficits)))
    effective_ws_per_turbine = base_ws * (1 - np.sqrt(total_deficit_sq))
    effective_ws_per_turbine = np.maximum(effective_ws_per_turbine, 0)
    
    # Target is the AVERAGE effective speed of the farm
    # (This represents the "True Potential" of the farm)
    avg_effective_ws = np.mean(effective_ws_per_turbine)
    
    return avg_effective_ws, adj

# --- DATA GENERATION LOOP ---
print("-> Generating Wind Data with Ramp Events...")
n_samples = 6000
t = np.linspace(0, 150, n_samples)

# Create complex wind profile (Target Speed)
base_ws_ser = 9 + 6 * np.sin(t) + 3 * np.sin(t*4.5) 
# Add Sharp Ramps
jumps = np.random.choice([-4, 4, 0], size=n_samples, p=[0.03, 0.03, 0.94])
base_ws_ser += jumps
base_ws_ser += np.random.normal(0, 0.5, n_samples)
base_ws_ser = np.clip(base_ws_ser, 0, CONFIG['max_wind_speed'])

base_wd_ser = (np.linspace(0, 720, n_samples) + np.random.normal(0, 15, n_samples)) % 360

X_list, y_list, adj_list, nwp_list = [], [], [], []

for i in range(n_samples):
    # Calculate True Effective Speed (Target)
    true_avg_speed, adj = simulate_ground_truth(base_ws_ser[i], base_wd_ser[i], turbine_coords)
    
    # Input Features (Simulated SCADA - Historical)
    nf = np.zeros((N_TURBINES, 3))
    nf[:, 0] = base_ws_ser[i] + np.random.normal(0, 0.5, N_TURBINES) # Measured Speed
    nf[:, 1] = np.cos(np.deg2rad(base_wd_ser[i]))
    nf[:, 2] = np.sin(np.deg2rad(base_wd_ser[i]))
    
    # NWP Forecast (Future Free Stream)
    # Important: NWP predicts Free Stream, while Target is Effective Speed (Waked)
    nwp_forecast = base_ws_ser[i] + np.random.normal(0, CONFIG['nwp_noise_level'], 1)
    
    X_list.append(nf)
    y_list.append(true_avg_speed) # Target: m/s
    adj_list.append(adj)
    nwp_list.append(nwp_forecast)

X_nodes = np.array(X_list)
A_dyn = np.array(adj_list)
y_total = np.array(y_list)

# Normalize Wind Speed to [0, 1] for Neural Net Stability
# We will Denormalize later for metrics
y_norm = y_total / CONFIG['max_wind_speed']
nwp_data = np.array(nwp_list) / CONFIG['max_wind_speed']
X_nodes[:, :, 0] = X_nodes[:, :, 0] / CONFIG['max_wind_speed'] # Normalize history too

# --- FEATURE ENGINEERING (DELTA) ---
def create_dataset(X, A, y, nwp, win):
    X_o, A_o, y_o, nwp_o, delta_o = [], [], [], [], []
    for i in range(len(X)-win):
        X_o.append(X[i:i+win])
        A_o.append(A[i+win-1])
        y_o.append(y[i+win])
        nwp_o.append(nwp[i+win])
        
        # Explicit Delta: Future NWP (Free) - Current Farm Avg (Waked)
        # This helps model learn the recovery trend
        current_avg = np.mean(X[i+win-1, :, 0])
        future_nwp = nwp[i+win][0]
        delta_val = future_nwp - current_avg
        delta_o.append([delta_val])

    return np.array(X_o), np.array(A_o), np.array(y_o), np.array(nwp_o), np.array(delta_o)

X_data, A_data, y_data, nwp_in, delta_in = create_dataset(
    X_nodes, A_dyn, y_norm, nwp_data, CONFIG['window_size']
)
nwp_in = nwp_in.reshape(-1, 1)

split = int(0.9 * len(X_data))
X_train, X_test = X_data[:split], X_data[split:]
A_train, A_test = A_data[:split], A_data[split:]
y_train, y_test = y_data[:split], y_data[split:]
nwp_train, nwp_test = nwp_in[:split], nwp_in[split:]
delta_train, delta_test = delta_in[:split], delta_in[split:]

print(f"-> Dataset Ready. Target is Wind Speed (m/s). Train size: {len(X_train)}")

# ==============================================================================
# SECTION 3: GRADIENT LOCKING MODEL (WIND SPEED ADAPTED)
# ==============================================================================

def gradient_locking_loss(y_true, y_pred):
    """
    Optimized Loss for Speed Prediction.
    Penalizes Slope Error heavily.
    """
    mse = K.mean(K.square(y_true - y_pred))
    
    # Correlation (Trend Matching)
    y_t_c = y_true - K.mean(y_true)
    y_p_c = y_pred - K.mean(y_pred)
    cov = K.mean(y_t_c * y_p_c)
    std_t = K.std(y_true) + 1e-6
    std_p = K.std(y_pred) + 1e-6
    corr = cov / (std_t * std_p)
    
    # Gradient Penalty
    grad_penalty = K.mean(K.abs(K.sign(y_true - K.mean(y_true)) - K.sign(y_pred - K.mean(y_pred))))
    
    return 1.0 * mse + 15.0 * (1.0 - corr) + 1.0 * grad_penalty

class DynamicGraphConv(tf.keras.layers.Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
    def build(self, input_shape):
        self.W = self.add_weight(shape=(3, self.out_features), initializer='glorot_uniform')
    def call(self, inputs):
        X, A = inputs
        X_trans = tf.matmul(X, self.W)
        A_exp = tf.expand_dims(A, 1)
        return tf.nn.relu(tf.matmul(A_exp, X_trans))

def build_wind_speed_model():
    # Inputs
    in_hist = Input(shape=(CONFIG['window_size'], N_TURBINES, 3), name='History')
    in_adj = Input(shape=(N_TURBINES, N_TURBINES), name='Adjacency')
    in_nwp = Input(shape=(1,), name='NWP_FreeStream') 
    in_delta = Input(shape=(1,), name='Delta_Trend') 
    
    # --- PHYSICAL BACKBONE (NWP BASELINE) ---
    # The simplest physics: Future Speed ~= NWP Forecast
    # We will subtract Wake Loss from this.
    p_baseline = Lambda(lambda x: x, name='Physics_Passthrough')(in_nwp)
    
    # --- AI BRANCH (WAKE & RAMP LEARNING) ---
    gnn = DynamicGraphConv(16)([in_hist, in_adj])
    flat = Reshape((CONFIG['window_size'], -1))(gnn)
    lstm_out = LSTM(64, return_sequences=False)(flat)
    
    # Context
    nwp_feat = Dense(32, activation='relu')(in_nwp)
    delta_feat = Dense(32, activation='relu')(in_delta)
    context = Concatenate()([lstm_out, nwp_feat, delta_feat])
    
    # Output 1: Wake Deficit Prediction (How much speed is lost?)
    # Sigmoid -> 0 to 1 range (representing 0% to 100% loss)
    # Usually wake loss is small (e.g. 0.1), so we scale it later or learn it directly
    wake_hidden = Dense(32, activation='relu')(context)
    wake_deficit = Dense(1, activation='sigmoid', name='Wake_Deficit')(wake_hidden)
    
    # Output 2: Ramp Correction (Additive)
    ramp_hidden = Dense(32, activation='relu')(context)
    ramp_correction = Dense(1, activation='linear', kernel_initializer='zeros', name='Ramp_Corr')(ramp_hidden)
    
    # --- FUSION LOGIC ---
    # Effective Speed = NWP (Free) - Wake Loss + Correction
    # Note: wake_deficit is normalized [0,1], representing absolute drop relative to scale
    
    # Apply Wake subtraction
    after_wake = Subtract()([p_baseline, wake_deficit])
    
    # Apply Gradient Correction
    final_out = Add()([after_wake, ramp_correction])
    
    # Clip to valid range [0, 1]
    final_out = Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(final_out)
    
    model = Model(inputs=[in_hist, in_adj, in_nwp, in_delta], outputs=final_out)
    model.compile(optimizer=Adam(CONFIG['learning_rate']), 
                  loss=gradient_locking_loss, 
                  metrics=['mae'])
    return model

model = build_wind_speed_model()

# ==============================================================================
# SECTION 4: TRAINING (SAMPLE WEIGHTING)
# ==============================================================================

# Calculate weights based on Wind Speed Delta (not Power)
delta_magnitude = np.abs(delta_train.flatten())
sample_weights = np.ones_like(delta_magnitude)

# Weighting logic adapted for normalized speed [0,1]
# A change of 0.04 (normalized) is approx 1 m/s (since max is 25)
sample_weights[delta_magnitude > 0.04] = 20.0 
sample_weights[(delta_magnitude > 0.02) & (delta_magnitude <= 0.04)] = 5.0

print("\n>>> STARTING TRAINING (Target: Effective Wind Speed)...")
history = model.fit(
    [X_train, A_train, nwp_train, delta_train], y_train,
    sample_weight=sample_weights,
    validation_split=0.1,
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=[
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(patience=8, factor=0.2)
    ],
    verbose=1
)

# --- EVALUATION ---
print("\n>>> CALCULATING METRICS (Denormalized)...")
# Denormalize back to m/s
y_pred_ms = model.predict([X_test, A_test, nwp_test, delta_test]).flatten() * CONFIG['max_wind_speed']
y_true_ms = y_test.flatten() * CONFIG['max_wind_speed']

def calculate_speed_metrics(y_t, y_p):
    rmse = np.sqrt(np.mean((y_t - y_p)**2))
    mae = np.mean(np.abs(y_t - y_p))
    
    # MAPE for Speed (avoid div by zero)
    mask = y_t > 1.0 # Only check where wind > 1 m/s
    mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100
    
    diff_t = np.diff(y_t)
    diff_p = np.diff(y_p)
    
    if np.std(diff_p) > 1e-6:
        ramp_corr = np.corrcoef(diff_t, diff_p)[0, 1]
    else: ramp_corr = 0
    
    sum_abs_diff_t = np.sum(np.abs(diff_t))
    if sum_abs_diff_t > 0:
        ramp_score = 1 - (np.sum(np.abs(diff_t - diff_p)) / sum_abs_diff_t)
    else: ramp_score = 0
    
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum((y_t - np.mean(y_t))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return rmse, mae, mape, ramp_score, r2, ramp_corr

rmse, mae, mape, ramp, r2, r_corr = calculate_speed_metrics(y_true_ms, y_pred_ms)

print("="*65)
print(f"BAC LIEU WIND SPEED FORECASTER - FINAL RESULTS")
print("="*65)
print(f"1. RMSE:        {rmse:.4f} m/s")
print(f"2. MAE:         {mae:.4f} m/s")
print(f"3. MAPE:        {mape:.2f} %")
print(f"4. Ramp Score:  {ramp:.4f}     (Expected > 0.2)")
print(f"5. R-Squared:   {r2:.4f}       (Target > 0.9)")
print(f"6. Ramp Corr:   {r_corr:.4f}   (Target > 0.6)")
print("="*65)

# Plot
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(y_true_ms[:200], 'k-', label='Actual Avg Speed (m/s)', alpha=0.8)
plt.plot(y_pred_ms[:200], 'g--', label='Predicted Avg Speed (m/s)', linewidth=2)
plt.title(f'Effective Wind Speed Forecasting (R2={r2:.3f})')
plt.ylabel('Speed (m/s)'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(np.diff(y_true_ms[:200]), 'k-', label='Actual Ramp', alpha=0.5)
plt.plot(np.diff(y_pred_ms[:200]), 'g-', label='Predicted Ramp', alpha=0.8)
plt.title(f'Wind Speed Ramp Capture (Score={ramp:.3f})')
plt.ylabel('Delta m/s'); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()