import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/Atlanta.csv')
df.columns = [c.strip() for c in df.columns]
df = df[df['Year'].notna() & (df['Year'] > 2000)].copy()
df['Year'] = df['Year'].astype(int)
df['Population'] = df['Estimate Population'].astype(str).str.replace(',','').astype(float)
df['NetMig']     = df['Net Mig'].astype(str).str.replace(',','').astype(float)
df['NatInc']     = df['Natural inc'].astype(str).str.replace(',','').astype(float)
df = df.sort_values('Year').reset_index(drop=True)

years  = df['Year'].values
pop    = df['Population'].values

print("Historical data loaded:")
print(df[['Year','Population','NetMig','NatInc']].to_string(index=False))

# ── 2. HOLT-WINTERS (DOUBLE EXPONENTIAL SMOOTHING) ───────────────────────────
# Manual implementation — no statsmodels needed
def holt_forecast(series, alpha, beta, n_ahead):
    """Holt's linear (double) exponential smoothing."""
    L = series[0]
    T = series[1] - series[0]
    fitted = []
    for t in range(len(series)):
        if t == 0:
            fitted.append(L + T)
            continue
        L_prev, T_prev = L, T
        L = alpha * series[t] + (1 - alpha) * (L_prev + T_prev)
        T = beta  * (L - L_prev) + (1 - beta) * T_prev
        fitted.append(L + T)
    forecasts = [L + i * T for i in range(1, n_ahead + 1)]
    return np.array(fitted), np.array(forecasts), L, T

# Grid search for best alpha/beta (minimize RMSE on historical)
best_rmse, best_a, best_b = 1e12, 0.5, 0.1
for a in np.arange(0.1, 1.0, 0.1):
    for b in np.arange(0.05, 0.5, 0.05):
        fitted, _, _, _ = holt_forecast(pop, a, b, 0)
        rmse = np.sqrt(np.mean((pop - fitted)**2))
        if rmse < best_rmse:
            best_rmse, best_a, best_b = rmse, a, b

print(f"\nHolt best params: alpha={best_a:.2f}, beta={best_b:.2f}, RMSE={best_rmse:,.0f}")
fitted_holt, fc_holt, L_final, T_final = holt_forecast(pop, best_a, best_b, 10)

# ── 3. LINEAR TREND (SIMPLE BENCHMARK) ───────────────────────────────────────
coeffs = np.polyfit(years, pop, 1)
poly   = np.poly1d(coeffs)
fc_years = np.arange(2025, 2035)
fc_linear = poly(fc_years)

# ── 4. COMPONENT-BASED SCENARIO BOUNDS ───────────────────────────────────────
# Use average annual growth from recent 5 years (2020-2024) for component cross-check
recent = df[df['Year'] >= 2020].copy()
avg_natinc = recent['NatInc'].mean()
avg_netmig = recent['NetMig'].mean()
base_pop   = pop[-1]  # 2024 = 6,411,149

# Scenarios: vary net migration (most volatile driver for Atlanta)
# Base: avg of last 5 years migration; High: +20k/yr; Low: -20k/yr
fc_component_base = []
fc_component_high = []
fc_component_low  = []
p_base = p_high = p_low = base_pop
for _ in range(10):
    p_base += avg_natinc + avg_netmig
    p_high += avg_natinc + avg_netmig + 20000
    p_low  += avg_natinc + avg_netmig - 20000
    fc_component_base.append(p_base)
    fc_component_high.append(p_high)
    fc_component_low.append(p_low)

fc_component_base = np.array(fc_component_base)
fc_component_high = np.array(fc_component_high)
fc_component_low  = np.array(fc_component_low)

# ── 5. FINAL FORECAST = blend Holt + Component (equal weight) ────────────────
fc_final = (fc_holt + fc_component_base) / 2
fc_high  = (fc_holt + fc_component_high) / 2
fc_low   = (fc_holt + fc_component_low)  / 2

print("\n=== 10-YEAR FORECAST (2025–2034) ===")
print(f"{'Year':>6} {'Base (M)':>12} {'Low (M)':>12} {'High (M)':>12}")
for i, yr in enumerate(fc_years):
    print(f"{yr:>6} {fc_final[i]/1e6:>12.4f} {fc_low[i]/1e6:>12.4f} {fc_high[i]/1e6:>12.4f}")

# ── 6. CHARTS ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#F8F9FA')

BLUE   = '#1A3A5C'
ORANGE = '#E07B2A'
GREEN  = '#2E8B57'
LGREY  = '#D0D5DD'

# ─ Chart 1: Historical + Forecast with uncertainty band ─
ax = axes[0]
ax.set_facecolor('white')
ax.plot(years, pop/1e6, color=BLUE, lw=2.5, marker='o', ms=5, label='Historical')
ax.plot(fc_years, fc_final/1e6, color=ORANGE, lw=2.5, marker='s', ms=5,
        linestyle='--', label='Base Forecast')
ax.fill_between(fc_years, fc_low/1e6, fc_high/1e6,
                color=ORANGE, alpha=0.18, label='Uncertainty Range')
ax.axvline(x=2024.5, color='grey', lw=1, linestyle=':')
ax.text(2024.6, pop.min()/1e6, '← Historical | Forecast →',
        fontsize=8, color='grey', va='bottom')
ax.set_title('Atlanta MSA: Population Forecast (2011–2034)',
             fontsize=13, fontweight='bold', color=BLUE, pad=12)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Population (Millions)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}M'))
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_xlim(2010, 2036)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

# ─ Chart 2: Annual Growth & Components ─
ax2 = axes[1]
ax2.set_facecolor('white')
ann_growth = np.diff(pop)
bar_years  = years[1:]
ax2.bar(bar_years, df['NatInc'].values[1:]/1000, color=GREEN,   label='Natural Increase (k)', alpha=0.85)
ax2.bar(bar_years, df['NetMig'].values[1:]/1000, bottom=df['NatInc'].values[1:]/1000,
        color=ORANGE, label='Net Migration (k)', alpha=0.85)
ax2.set_title('Components of Population Change\n(Natural Increase + Net Migration)',
              fontsize=13, fontweight='bold', color=BLUE, pad=12)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel("Annual Change (thousands)", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
for spine in ['top','right']:
    ax2.spines[spine].set_visible(False)

plt.tight_layout(pad=3)
plt.savefig('/mnt/user-data/outputs/atlanta_forecast.png', dpi=150, bbox_inches='tight')
print("\nChart saved.")

# ── 7. EXPORT TABLE ──────────────────────────────────────────────────────────
hist_table = pd.DataFrame({
    'Year': years,
    'Type': 'Historical',
    'Population': pop.astype(int),
    'Population_M': (pop/1e6).round(4),
    'Net_Migration': df['NetMig'].values.astype(int),
    'Natural_Increase': df['NatInc'].values.astype(int)
})
fc_table = pd.DataFrame({
    'Year': fc_years,
    'Type': 'Forecast',
    'Population': fc_final.astype(int),
    'Population_M': (fc_final/1e6).round(4),
    'Net_Migration': int(avg_netmig),
    'Natural_Increase': int(avg_natinc)
})
fc_table['Low_Scenario']  = fc_low.astype(int)
fc_table['High_Scenario'] = fc_high.astype(int)
full_table = pd.concat([hist_table, fc_table], ignore_index=True)
full_table.to_csv('/mnt/user-data/outputs/atlanta_population_table.csv', index=False)
print("Table saved.")
print(f"\nForecast 2034 (Base): {fc_final[-1]/1e6:.3f}M")
print(f"Forecast 2034 (Low):  {fc_low[-1]/1e6:.3f}M")
print(f"Forecast 2034 (High): {fc_high[-1]/1e6:.3f}M")
