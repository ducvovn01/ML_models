import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. SETUP STYLE
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 2. LOAD & PROCESS DATA
# ==========================================
try:
    df = pd.read_csv('Dataset15years.csv')
    df.columns = ['time', 'Temp', 'WindSpeed_kmh', 'WindDir', 'Pressure', 'Humidity']
    df['time'] = pd.to_datetime(df['time'])
except:
    # Create Dummy Data if file missing (For demonstration)
    print("File not found. Generating dummy data for demo...")
    dates = pd.date_range(start='2020-01-01', end='2020-12-31 23:00', freq='H')
    # Simulate: Dry season (months 1,2,3,11,12) has stronger wind than Rainy (5-10)
    month_factor = np.where((dates.month >= 5) & (dates.month <= 10), 5.0, 8.5) 
    day_factor = 2 * np.sin((dates.hour - 8) * 2 * np.pi / 24) # Day/Night cycle
    ws = month_factor + day_factor + np.random.normal(0, 1.5, len(dates))
    df = pd.DataFrame({'time': dates, 'WindSpeed_kmh': ws * 3.6})

# Convert to m/s
df['Wind_Speed_ms'] = df['WindSpeed_kmh'] / 3.6
df['Month'] = df['time'].dt.month
df['Hour'] = df['time'].dt.hour

# 3. DEFINE SEASONS (BAC LIEU CONTEXT)
# ==========================================
def get_season(month):
    # Rainy: May (5) to October (10)
    if 5 <= month <= 10:
        return 'Rainy Season (May - Oct)'
    # Dry: November (11) to April (4)
    else:
        return 'Dry Season (Nov - Apr)'

df['Season'] = df['Month'].apply(get_season)

# 4. PLOTTING (DIURNAL CYCLE)
# ==========================================
fig, ax = plt.subplots()

# Use Seaborn Lineplot
# It automatically calculates the Mean (solid line) and Confidence Interval (shaded area)
sns.lineplot(
    data=df, 
    x='Hour', 
    y='Wind_Speed_ms', 
    hue='Season', 
    palette={'Dry Season (Nov - Apr)': '#d32f2f', 'Rainy Season (May - Oct)': '#1976d2'},
    linewidth=2.5,
    ax=ax
)

# 5. CUSTOMIZE LABELS (ENGLISH)
# ==========================================
plt.title("Typical Daily Wind Profile: Dry vs. Rainy Season (Bac Lieu)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Hour of Day (00:00 - 23:00)", fontsize=12)
plt.ylabel("Average Wind Speed (m/s)", fontsize=12)

# Set X-axis ticks to show all hours
plt.xticks(range(0, 24, 2))
plt.xlim(0, 23)

# Add Annotations for Context
# Note: Coordinates (x, y) might need adjustment based on your real data range
y_max = df.groupby(['Season', 'Hour'])['Wind_Speed_ms'].mean().max()
plt.text(14, y_max + 0.5, 'Peak Wind (Afternoon)', ha='center', fontsize=10, style='italic', color='gray')

# Add a horizontal line for Cut-in speed (approx 3.5 m/s)
plt.axhline(y=3.5, color='gray', linestyle='--', alpha=0.7)
plt.text(0.5, 3.6, 'Cut-in Speed (3.5 m/s)', fontsize=9, color='gray')

# Legend
plt.legend(title='Season', title_fontsize='11', fontsize='10', loc='upper left')

plt.tight_layout()
plt.show()