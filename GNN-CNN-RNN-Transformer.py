

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import (Input, Dense, LSTM, Reshape, Concatenate, 
                                     Add, Multiply, Lambda, Activation, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

# --- 1. CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
np.random.seed(42)
tf.random.set_seed(42)

CONFIG = {
    'window_size': 24,
    'batch_size': 64,
    'epochs': 100,            # Increased epochs to let Booster learn
    'learning_rate': 0.0001,  # Lower LR for stability with high weights
    
    # SPECS
    'rated_power_mw': 1.6,
    'rotor_diameter': 82.5,
    'cut_in_speed': 3.5,
    'rated_speed': 12.0,
    'cut_out_speed': 25.0,
    'wake_decay_k': 0.075, 
    'nwp_noise_level': 0.15   # Slightly cleaner NWP to help correlation
}

print(f"Starting Model...")

# ==============================================================================
# SECTION 2: EXACT LAYOUT (FISHBONE)
# ==============================================================================

def generate_bac_lieu_fishbone_layout():
    coords = []
    # 4 rows of 8, 3 rows of 7, 3 rows of 3
    rows_config = [8, 8, 8, 8, 7, 7, 7, 3, 3, 3] 
    
    lateral_spacing = 600
    longitudinal_spacing = 500
    
    for row_idx, n_turbines in enumerate(rows_config):
        x_base = row_idx * lateral_spacing
        for t_idx in range(n_turbines):
            x_pos = x_base + np.random.uniform(-10, 10)
            y_pos = t_idx * longitudinal_spacing + np.random.uniform(-10, 10)
            coords.append([x_pos, y_pos])
            
    return np.array(coords)

turbine_coords = generate_bac_lieu_fishbone_layout()
N_TURBINES = len(turbine_coords)
TOTAL_CAPACITY_MW = N_TURBINES * CONFIG['rated_power_mw']

print(f"-> Layout: {N_TURBINES} Turbines. Max Capacity: {TOTAL_CAPACITY_MW:.2f} MW")

# ==============================================================================
# SECTION 3: PHYSICS ENGINE
# ==============================================================================

def compute_jensen_adjacency(wind_dir_deg, coords, D, k):
    n = len(coords)
    adj = np.zeros((n, n))
    theta = np.radians(90 - wind_dir_deg)
    rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
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
    adj = compute_jensen_adjacency(base_wd, coords, CONFIG['rotor_diameter'], CONFIG['wake_decay_k'])
    total_deficit_sq = np.sum(adj**2, axis=1)
    effective_ws = base_ws * (1 - np.sqrt(total_deficit_sq))
    effective_ws = np.maximum(effective_ws, 0)
    
    power_mw = np.zeros(len(coords))
    mask_ramp = (effective_ws >= CONFIG['cut_in_speed']) & (effective_ws < CONFIG['rated_speed'])
    v_range = CONFIG['rated_speed'] - CONFIG['cut_in_speed']
    power_mw[mask_ramp] = CONFIG['rated_power_mw'] * ((effective_ws[mask_ramp] - CONFIG['cut_in_speed']) / v_range)**3
    mask_rated = (effective_ws >= CONFIG['rated_speed']) & (effective_ws < CONFIG['cut_out_speed'])
    power_mw[mask_rated] = CONFIG['rated_power_mw']
    
    return np.sum(power_mw), adj

# --- DATA GEN (Strong Ramps) ---
print("-> Generating High-Contrast Ramp Data...")
n_samples = 6500 
t = np.linspace(0, 160, n_samples)
base_ws_ser = 8 + 7 * np.sin(t) + 4 * np.cos(t*2.5) # Dynamic base

# Inject systematic ramps (steps) to force learning
jumps = np.random.choice([-5, 5, 0], size=n_samples, p=[0.04, 0.04, 0.92])
base_ws_ser += jumps
base_ws_ser += np.random.normal(0, 0.3, n_samples)
base_ws_ser = np.clip(base_ws_ser, 0, 28)
base_wd_ser = (np.linspace(0, 720, n_samples) + np.random.normal(0, 10, n_samples)) % 360

X_list, y_list, adj_list, nwp_list = [], [], [], []
for i in range(n_samples):
    p_true, adj = simulate_ground_truth(base_ws_ser[i], base_wd_ser[i], turbine_coords)
    nf = np.zeros((N_TURBINES, 3))
    nf[:, 0] = base_ws_ser[i] + np.random.normal(0, 0.5, N_TURBINES)
    nf[:, 1] = np.cos(np.deg2rad(base_wd_ser[i]))
    nf[:, 2] = np.sin(np.deg2rad(base_wd_ser[i]))
    nwp_forecast = base_ws_ser[i] + np.random.normal(0, CONFIG['nwp_noise_level'], 1)
    
    X_list.append(nf); y_list.append(p_true); adj_list.append(adj); nwp_list.append(nwp_forecast)

X_nodes = np.array(X_list); A_dyn = np.array(adj_list); y_total = np.array(y_list); nwp_data = np.array(nwp_list)
y_norm = y_total / TOTAL_CAPACITY_MW

def create_dataset_with_trend(X, A, y, nwp, win):
    X_o, A_o, y_o, nwp_o, delta_o, trend_o = [], [], [], [], [], []
    for i in range(len(X)-win):
        X_o.append(X[i:i+win]); A_o.append(A[i+win-1]); y_o.append(y[i+win]); nwp_o.append(nwp[i+win])
        
        # 1. Delta: Future NWP vs Current Mean
        curr_wind = np.mean(X[i+win-1, :, 0])
        future_wind = nwp[i+win][0]
        delta_val = future_wind - curr_wind
        delta_o.append([delta_val])
        
        # 2. Trend: NWP(t) - NWP(t-1) (Instant derivative of forecast)
        if i > 0:
            trend_val = nwp[i+win][0] - nwp[i+win-1][0]
        else:
            trend_val = 0.0
        trend_o.append([trend_val])

    return np.array(X_o), np.array(A_o), np.array(y_o), np.array(nwp_o), np.array(delta_o), np.array(trend_o)

X_data, A_data, y_data, nwp_in, delta_in, trend_in = create_dataset_with_trend(X_nodes, A_dyn, y_norm, nwp_data, CONFIG['window_size'])
nwp_in = nwp_in.reshape(-1, 1)

split = int(0.9 * len(X_data))
X_train, X_test = X_data[:split], X_data[split:]
A_train, A_test = A_data[:split], A_data[split:]
y_train, y_test = y_data[:split], y_data[split:]
nwp_train, nwp_test = nwp_in[:split], nwp_in[split:]
delta_train, delta_test = delta_in[:split], delta_in[split:]
trend_train, trend_test = trend_in[:split], trend_in[split:]

# ==============================================================================
# SECTION 4: DUAL-STREAM MODEL WITH BOOSTER
# ==============================================================================

def ultimate_gradient_loss(y_true, y_pred):
    """
    Punishes 'wrong direction' severely.
    """
    mse = K.mean(K.square(y_true - y_pred))
    
    # Correlation Check
    y_t_c = y_true - K.mean(y_true); y_p_c = y_pred - K.mean(y_pred)
    cov = K.mean(y_t_c * y_p_c); std_t = K.std(y_true) + 1e-6; std_p = K.std(y_pred) + 1e-6
    corr = cov / (std_t * std_p)
    
    # Sign Penalty: If True increases but Pred decreases (or vice versa) -> HUGE penalty
    diff_true = y_true[1:] - y_true[:-1]
    diff_pred = y_pred[1:] - y_pred[:-1]
    # If product is negative, signs are opposite
    sign_mismatch = K.mean(K.relu(-1.0 * diff_true * diff_pred)) 
    
    # Standard Gradient Penalty
    grad_penalty = K.mean(K.abs(K.sign(diff_true) - K.sign(diff_pred)))
    
    # High weights for shape-related terms
    return 1.0 * mse + 30.0 * (1.0 - corr) + 10.0 * grad_penalty + 10.0 * sign_mismatch

class LearnableMultiplier(Layer):
    """Scales the ramp correction by a learnable factor (initially > 1)."""
    def __init__(self, init_value=2.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
    def build(self, input_shape):
        self.kernel = self.add_weight(name='ramp_scale', shape=(1,), 
                                      initializer=tf.constant_initializer(self.init_value),
                                      trainable=True)
    def call(self, inputs):
        return inputs * self.kernel

class CubicPowerLayer(Layer):
    def call(self, inputs):
        v_eff = tf.nn.relu(inputs - CONFIG['cut_in_speed']) 
        v_range = CONFIG['rated_speed'] - CONFIG['cut_in_speed']
        ratio = v_eff / v_range
        p_cubic = tf.pow(ratio, 3)
        return tf.minimum(p_cubic, 1.0)

class DynamicGraphConv(Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs); self.out_features = out_features
    def build(self, input_shape):
        self.W = self.add_weight(shape=(3, self.out_features), initializer='glorot_uniform')
    def call(self, inputs):
        X, A = inputs
        X_trans = tf.matmul(X, self.W); A_exp = tf.expand_dims(A, 1)
        return tf.nn.relu(tf.matmul(A_exp, X_trans))

def build_ultimate_model():
    in_hist = Input(shape=(CONFIG['window_size'], N_TURBINES, 3))
    in_adj = Input(shape=(N_TURBINES, N_TURBINES))
    in_nwp = Input(shape=(1,)); in_delta = Input(shape=(1,)); in_trend = Input(shape=(1,))
    
    # 1. Physics Base
    p_baseline = CubicPowerLayer()(in_nwp)
    
    # 2. Spatiotemporal Feature
    gnn = DynamicGraphConv(16)([in_hist, in_adj])
    flat = Reshape((CONFIG['window_size'], -1))(gnn)
    lstm_out = LSTM(64, return_sequences=False)(flat)
    
    # 3. Context
    nwp_feat = Dense(32, activation='relu')(in_nwp)
    
    # 4. EXPLICIT TREND BRANCH (The "Fast Lane")
    # Combines Delta (Difference from now) and Trend (NWP slope)
    trend_feat = Concatenate()([in_delta, in_trend])
    trend_feat = Dense(64, activation='relu')(trend_feat)
    
    context = Concatenate()([lstm_out, nwp_feat, trend_feat])
    
    # Output 1: Efficiency (Slow changing)
    eff_factor = Dense(1, activation='sigmoid', bias_initializer='ones', name='Eff')(Dense(32, activation='relu')(context))
    
    # Output 2: Ramp Correction (Fast changing)
    # Tanh allows pushing up or down
    ramp_raw = Dense(1, activation='tanh', name='Ramp_Raw')(Dense(64, activation='relu')(context))
    
    # BOOSTER: Amplify the correction
    ramp_boosted = LearnableMultiplier(init_value=2.0)(ramp_raw) 
    
    # Fusion
    base_eff = Multiply()([p_baseline, eff_factor])
    final_out = Add()([base_eff, ramp_boosted])
    final_out = Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(final_out)
    
    model = Model(inputs=[in_hist, in_adj, in_nwp, in_delta, in_trend], outputs=final_out)
    model.compile(optimizer=Adam(CONFIG['learning_rate']), loss=ultimate_gradient_loss, metrics=['mae'])
    return model

model = build_ultimate_model()

# ==============================================================================
# SECTION 5: TRAINING
# ==============================================================================

print("\n>>> STARTING TRAINING (Ultimate Mode)...")
delta_mag = np.abs(delta_train.flatten())
weights = np.ones_like(delta_mag)
# Massive weights for events
weights[delta_mag > 1.0] = 40.0 
weights[(delta_mag > 0.5) & (delta_mag <= 1.0)] = 10.0

history = model.fit(
    [X_train, A_train, nwp_train, delta_train, trend_train], y_train,
    sample_weight=weights, validation_split=0.1,
    epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
    callbacks=[EarlyStopping(patience=20, restore_best_weights=True), ReduceLROnPlateau(patience=5)],
    verbose=1
)

print("\n>>> FINAL EVALUATION...")
y_pred_mw = model.predict([X_test, A_test, nwp_test, delta_test, trend_test]).flatten() * TOTAL_CAPACITY_MW
y_true_mw = y_test.flatten() * TOTAL_CAPACITY_MW

def calculate_metrics(y_t, y_p):
    rmse = np.sqrt(np.mean((y_t - y_p)**2))
    mae = np.mean(np.abs(y_t - y_p))
    mask = y_t > 1.0 
    mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.sum(mask)>0 else 0
    
    # sMAPE
    num = np.abs(y_t - y_p)
    den = (np.abs(y_t) + np.abs(y_p)) / 2 + 1e-8
    smape = np.mean(num / den) * 100
    
    # Ramp Analysis
    diff_t = np.diff(y_t); diff_p = np.diff(y_p)
    
    if np.std(diff_p) > 1e-6:
        ramp_corr = np.corrcoef(diff_t, diff_p)[0, 1]
    else: ramp_corr = 0
    
    sum_abs_diff_t = np.sum(np.abs(diff_t))
    ramp_score = 1 - (np.sum(np.abs(diff_t - diff_p)) / sum_abs_diff_t) if sum_abs_diff_t > 0 else 0
    
    return rmse, mae, mape, smape, ramp_score, ramp_corr

rmse, mae, mape, smape, ramp, r_corr = calculate_metrics(y_true_mw, y_pred_mw)

print("="*65)
print(f"BAC LIEU ULTIMATE FISHBONE MODEL")
print("="*65)
print(f"1. RMSE:        {rmse:.4f} MW")
print(f"2. MAE:         {mae:.4f} MW")
print(f"3. MAPE:        {mape:.2f} %")
print(f"4. sMAPE:       {smape:.2f} %")
print(f"5. Ramp Score:  {ramp:.4f}     ")
print(f"6. Ramp Corr:   {r_corr:.4f}   ")
print("="*65)

plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(y_true_mw[:200], 'k-', label='Actual', alpha=0.7)
plt.plot(y_pred_mw[:200], 'r--', label='Predicted', linewidth=2)
plt.title(f"Forecast Overview (RMSE={rmse:.2f})")
plt.ylabel("MW"); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(np.diff(y_true_mw[:200]), 'k-', label='Actual Slope', alpha=0.5)
plt.plot(np.diff(y_pred_mw[:200]), 'g-', label='Pred Slope', alpha=0.8)
plt.title(f"Slope Analysis (Ramp Score={ramp:.4f})")
plt.ylabel("Delta MW"); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.scatter(np.diff(y_true_mw[:500]), np.diff(y_pred_mw[:500]), alpha=0.3, c='blue')
plt.plot([-20, 20], [-20, 20], 'r--')
plt.title(f"Correlation Scatter (Corr={r_corr:.4f})")
plt.xlabel("Actual Delta"); plt.ylabel("Predicted Delta")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()