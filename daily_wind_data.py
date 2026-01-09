import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 6) 

try:
    df = pd.read_csv('Dataset15years.csv')
    df.columns = ['time', 'Temp', 'WindSpeed_kmh', 'WindDir', 'Pressure', 'Humidity']
    df['time'] = pd.to_datetime(df['time'])
    
    
    available_years = df['time'].dt.year.unique()
    
    
    selected_year = np.random.choice(available_years)
    print(f">>> Randomly selected: {selected_year} to draw.")
    
except:
    print("No dataset found. Generating synthetic data for demonstration.")
    selected_year = 2022
    dates = pd.date_range(start=f'{selected_year}-01-01', end=f'{selected_year}-12-31 23:00', freq='H')
    
    trend = 6 + 3 * np.cos((dates.dayofyear - 15) * 2 * np.pi / 365) 
    noise = np.random.normal(0, 2.0, len(dates))
    ws_kmh = (trend + noise) * 3.6
    
    ws_kmh = np.clip(ws_kmh, 0, None)
    df = pd.DataFrame({'time': dates, 'WindSpeed_kmh': ws_kmh})


df_year = df[df['time'].dt.year == selected_year].copy()

df_year['Wind_Speed_ms'] = df_year['WindSpeed_kmh'] / 3.6


daily_data = df_year.set_index('time').resample('D')['Wind_Speed_ms'].mean()


rolling_7d = daily_data.rolling(window=7, center=True).mean()


# ==========================================
fig, ax = plt.subplots()


ax.plot(daily_data.index, daily_data, 
        color='#455a64', alpha=0.6, linewidth=1, label='Daily Average Wind Speed')


ax.plot(daily_data.index, rolling_7d, 
        color='#ff5722', linewidth=2.5, label='7-Day Moving Average (Trend)')


ax.axhline(y=6, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(daily_data.index[0], 6.2, 'Commercial Threshold (6 m/s)', color='green', fontsize=10, fontweight='bold')


# ==========================================
plt.title(f"Full Year Wind Speed Fluctuation - Year {selected_year} (Bac Lieu)", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Wind Speed (m/s)", fontsize=12)
plt.xlabel("Date", fontsize=12)


ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Jan, Feb, Mar...


start_date = pd.Timestamp(f'{selected_year}-01-01')
end_date = pd.Timestamp(f'{selected_year}-12-31')


ax.axvspan(pd.Timestamp(f'{selected_year}-01-01'), pd.Timestamp(f'{selected_year}-03-31'), color='orange', alpha=0.1)

ax.axvspan(pd.Timestamp(f'{selected_year}-11-01'), pd.Timestamp(f'{selected_year}-12-31'), color='orange', alpha=0.1)

plt.text(pd.Timestamp(f'{selected_year}-02-15'), daily_data.max(), 'High Wind Season', ha='center', color='orange', fontweight='bold')

plt.legend(loc='upper center', frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.show()