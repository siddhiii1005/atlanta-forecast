import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv('Atlanta.csv')
df.columns = [c.strip() for c in df.columns]
df = df[df['Year'].notna() & (df['Year'] > 2000)].copy()
df['Year'] = df['Year'].astype(int)
df['Population'] = df['Estimate Population'].astype(float)
df['NetMig'] = df['Net Mig'].astype(float)
df['NatInc'] = df['Natural inc'].astype(float)
df = df.sort_values('Year').reset_index(drop=True)

years = df['Year'].values
pop = df['Population'].values


def holt(series, alpha, beta, n_ahead):
    # double exponential smoothing - models level and trend separately
    L = series[0]
    T = series[1] - series[0]
    fitted = []
    for t in range(len(series)):
        if t == 0:
            fitted.append(L + T)
            continue
        L_prev, T_prev = L, T
        L = alpha * series[t] + (1 - alpha) * (L_prev + T_prev)
        T = beta * (L - L_prev) + (1 - beta) * T_prev
        fitted.append(L + T)
    fc = np.array([L + i * T for i in range(1, n_ahead + 1)])
    return np.array(fitted), fc


# grid search over alpha/beta to minimise in-sample RMSE
best_rmse = 1e12
best_a, best_b = 0.5, 0.1
for a in np.arange(0.1, 1.0, 0.1):
    for b in np.arange(0.05, 0.5, 0.05):
        fitted, _ = holt(pop, a, b, 0)
        rmse = np.sqrt(np.mean((pop - fitted) ** 2))
        if rmse < best_rmse:
            best_rmse, best_a, best_b = rmse, a, b

fitted_holt, fc_holt = holt(pop, best_a, best_b, 10)
fc_years = np.arange(2025, 2035)

# component model: P(t) = P(t-1) + natural increase + net migration
# using 2020-2024 averages rather than full series - pre-2020 birth rates were higher
# and unlikely to reflect the next decade given population aging
recent = df[df['Year'] >= 2020]
avg_natinc = recent['NatInc'].mean()
avg_netmig = recent['NetMig'].mean()

p_base = p_high = p_low = pop[-1]
fc_comp_base, fc_comp_high, fc_comp_low = [], [], []
for _ in range(10):
    p_base += avg_natinc + avg_netmig
    p_high += avg_natinc + avg_netmig + 20000
    p_low += avg_natinc + avg_netmig - 20000
    fc_comp_base.append(p_base)
    fc_comp_high.append(p_high)
    fc_comp_low.append(p_low)

fc_comp_base = np.array(fc_comp_base)
fc_comp_high = np.array(fc_comp_high)
fc_comp_low = np.array(fc_comp_low)

# blend: equal weight between holt and component model
fc_base = (fc_holt + fc_comp_base) / 2
fc_high = (fc_holt + fc_comp_high) / 2
fc_low = (fc_holt + fc_comp_low) / 2

print(f"Best params: alpha={best_a:.2f}, beta={best_b:.2f}, RMSE={best_rmse:,.0f}")
print(f"\n{'Year':>6} {'Base':>12} {'Low':>12} {'High':>12}")
for i, yr in enumerate(fc_years):
    print(f"{yr:>6} {fc_base[i]:>12,.0f} {fc_low[i]:>12,.0f} {fc_high[i]:>12,.0f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#F8F9FA')

BLUE = '#1A3A5C'
ORANGE = '#E07B2A'
GREEN = '#2E8B57'

ax1.set_facecolor('white')
ax1.plot(years, pop / 1e6, color=BLUE, lw=2.5, marker='o', ms=5, label='Historical')
ax1.plot(fc_years, fc_base / 1e6, color=ORANGE, lw=2.5, marker='s', ms=5,
         linestyle='--', label='Base Forecast')
ax1.fill_between(fc_years, fc_low / 1e6, fc_high / 1e6,
                 color=ORANGE, alpha=0.18, label='Uncertainty Range')
ax1.axvline(x=2024.5, color='grey', lw=1, linestyle=':')
ax1.text(2024.7, pop.min() / 1e6 + 0.02, 'Forecast', fontsize=8, color='grey')
ax1.set_title('Atlanta MSA: Population Forecast (2011-2034)',
              fontsize=12, fontweight='bold', color=BLUE, pad=10)
ax1.set_xlabel('Year')
ax1.set_ylabel('Population (Millions)')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}M'))
ax1.legend(fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_xlim(2010, 2036)
for s in ['top', 'right']:
    ax1.spines[s].set_visible(False)

ax2.set_facecolor('white')
bar_years = years[1:]
nat = df['NatInc'].values[1:] / 1000
mig = df['NetMig'].values[1:] / 1000
ax2.bar(bar_years, nat, color=GREEN, label='Natural Increase (k)', alpha=0.85)
ax2.bar(bar_years, mig, bottom=nat, color=ORANGE, label='Net Migration (k)', alpha=0.85)
ax2.set_title('Components of Population Change',
              fontsize=12, fontweight='bold', color=BLUE, pad=10)
ax2.set_xlabel('Year')
ax2.set_ylabel('Annual Change (thousands)')
ax2.legend(fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
for s in ['top', 'right']:
    ax2.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig('atlanta_forecast.png', dpi=150, bbox_inches='tight')
print("\nChart saved to atlanta_forecast.png")

hist_out = pd.DataFrame({
    'Year': years,
    'Type': 'Historical',
    'Population': pop.astype(int),
    'Natural_Increase': df['NatInc'].values.astype(int),
    'Net_Migration': df['NetMig'].values.astype(int),
    'Low_Scenario': np.nan,
    'High_Scenario': np.nan
})
fc_out = pd.DataFrame({
    'Year': fc_years,
    'Type': 'Forecast',
    'Population': fc_base.astype(int),
    'Natural_Increase': int(avg_natinc),
    'Net_Migration': int(avg_netmig),
    'Low_Scenario': fc_low.astype(int),
    'High_Scenario': fc_high.astype(int)
})

pd.concat([hist_out, fc_out], ignore_index=True).to_csv(
    'atlanta_population_table.csv', index=False
)
print("Table saved to atlanta_population_table.csv")
