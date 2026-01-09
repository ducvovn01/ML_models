import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# Cấu hình giao diện đẹp
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# 1. ĐỌC DỮ LIỆU
# ==========================================
try:
    df = pd.read_csv('Dataset15years.csv')
    df.columns = ['time', 'Temp', 'WindSpeed_kmh', 'WindDir', 'Pressure', 'Humidity']
    df['time'] = pd.to_datetime(df['time'])
except:
    print("Dataset not found. Using dummy data for demonstration.")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='H')
    # Giả lập quy luật gió Bạc Liêu
    ws = 6 + 3 * np.cos((dates.dayofyear - 15) * 2 * np.pi / 365) + np.random.normal(0, 2, len(dates))
    df = pd.DataFrame({'time': dates, 'WindSpeed_kmh': ws * 3.6})

# 2. XỬ LÝ DỮ LIỆU
# ==========================================
target_year = 2023 
df_year = df[df['time'].dt.year == target_year].copy()

if len(df_year) == 0:
    target_year = df['time'].dt.year.iloc[0]
    df_year = df[df['time'].dt.year == target_year].copy()

# Convert km/h -> m/s
df_year['WindSpeed_ms'] = df_year['WindSpeed_kmh'] / 3.6

# Resample Daily Mean
daily_df = df_year.set_index('time').resample('D')['WindSpeed_ms'].mean()

# Rolling Mean (30 days)
rolling_mean = daily_df.rolling(window=30, center=True).mean()

# 3. VẼ BIỂU ĐỒ (CLEAN VERSION)
# ==========================================
fig, ax = plt.subplots()

# 1. Daily Data Points
ax.plot(daily_df.index, daily_df, 
        color='#b0bec5', alpha=0.6, linewidth=1, 
        label='Daily Average')

# 2. Trend Line
ax.plot(rolling_mean.index, rolling_mean, 
        color='#0277bd', linewidth=3, 
        label='Monthly Trend (30-day Moving Avg)')

# 3. Highlight Good Wind Zone
ax.fill_between(daily_df.index, daily_df, 6, where=(daily_df >= 6), 
                interpolate=True, color='#4caf50', alpha=0.2, 
                label='Profitable Wind Zone (>6 m/s)')

# --- TITLES & LABELS ---
plt.title(f"Wind Speed Variation in Bac Lieu - Year {target_year}", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Wind Speed (m/s)", fontsize=12)
plt.xlabel("Month of Year", fontsize=12)

# --- AXIS FORMATTING ---
# Chuyển đổi số tháng sang tên tiếng Anh (Jan, Feb, Mar...)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) 

# --- LEGEND ---
# Đặt legend gọn gàng bên dưới
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=11)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()