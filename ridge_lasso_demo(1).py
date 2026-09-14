import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Synthetic California-style dataset (no download needed) ──────────
np.random.seed(42)
n = 2000

MedInc      = np.random.exponential(3.5, n)
HouseAge    = np.random.uniform(1, 52, n)
AveRooms    = np.random.normal(5.4, 2.1, n).clip(1)
AveBedrms   = AveRooms * np.random.uniform(0.18, 0.25, n)
Population  = np.random.exponential(1400, n)
AveOccup    = np.random.normal(3.1, 1.5, n).clip(1)
Latitude    = np.random.uniform(32, 42, n)
Longitude   = np.random.uniform(-124, -114, n)

noise = np.random.normal(0, 0.5, n)

y = (
    0.45 * MedInc
    + 0.008 * HouseAge
    + 0.12  * AveRooms
    - 0.09  * AveBedrms
    - 0.00001 * Population
    - 0.04  * AveOccup
    - 0.42  * Latitude
    + 0.35  * Longitude
    + noise
)
y = (y - y.min()) / (y.max() - y.min()) * 5 + 0.3   # scale to ~[0.3, 5.3]

feature_names = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
X = np.column_stack([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ── 2. Fit Models ─────────────────────────────────────────────────────────
ols   = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=10).fit(X_train, y_train)
lasso = Lasso(alpha=0.05).fit(X_train, y_train)

# ── 3. Coefficient paths ──────────────────────────────────────────────────
alphas = np.logspace(-3, 3, 120)
ridge_coefs = np.array([Ridge(alpha=a).fit(X_train, y_train).coef_ for a in alphas])
lasso_coefs = np.array([Lasso(alpha=a, max_iter=10000).fit(X_train, y_train).coef_ for a in alphas])

# ── 4. Metrics ────────────────────────────────────────────────────────────
def metrics(model, Xt, yt):
    yp = model.predict(Xt)
    return r2_score(yt, yp), np.sqrt(mean_squared_error(yt, yp))

ols_r2,   ols_rmse   = metrics(ols,   X_test, y_test)
ridge_r2, ridge_rmse = metrics(ridge, X_test, y_test)
lasso_r2, lasso_rmse = metrics(lasso, X_test, y_test)

# ── 5. Styling ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#0F1117',
    'axes.facecolor': '#181B26',
    'axes.labelcolor': '#B0B3C8',
    'xtick.color': '#6E718A',
    'ytick.color': '#6E718A',
    'text.color': '#C8C9D4',
    'grid.color': '#252836',
    'grid.linewidth': 0.5,
    'axes.titlecolor': '#E8E9F5',
    'axes.titlesize': 10.5,
    'axes.titleweight': 'bold',
    'axes.spines.left':True,
    'axes.spines.bottom':True,
    'axes.edgecolor':'#2E3144',
})

BLUE   = '#4A9EFF'
CORAL  = '#FF6B6B'
GREEN  = '#5ECFA0'
AMBER  = '#FFB347'

fig = plt.figure(figsize=(17, 12))
fig.patch.set_facecolor('#0F1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

fig.suptitle('Ridge  vs  Lasso Regression  ·  California Housing  (Synthetic)',
             fontsize=15, fontweight='bold', color='#E8E9F5', y=0.97)

colors_path = plt.cm.cool(np.linspace(0.1, 0.9, len(feature_names)))

# ─── Plot 1: Ridge coef path ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for i, (feat, col) in enumerate(zip(feature_names, colors_path)):
    ax1.plot(np.log10(alphas), ridge_coefs[:, i], color=col, lw=1.5, label=feat)
ax1.axvline(np.log10(10), color=AMBER, lw=1.3, ls='--', alpha=0.8, label='α=10 used')
ax1.axhline(0, color='#FFFFFF', lw=0.4, alpha=0.2)
ax1.set_title('Ridge — Coefficient Path')
ax1.set_xlabel('log₁₀(α)'); ax1.set_ylabel('Coefficient')
ax1.legend(fontsize=6.5, loc='upper right', framealpha=0.12, ncol=2)
ax1.grid(True, alpha=0.35)
ax1.text(0.02, 0.05, 'All coefs shrink but NEVER reach 0', transform=ax1.transAxes,
         fontsize=7.5, color=BLUE, style='italic')

# ─── Plot 2: Lasso coef path ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for i, (feat, col) in enumerate(zip(feature_names, colors_path)):
    ax2.plot(np.log10(alphas), lasso_coefs[:, i], color=col, lw=1.5, label=feat)
ax2.axvline(np.log10(0.05), color=AMBER, lw=1.3, ls='--', alpha=0.8, label='α=0.05 used')
ax2.axhline(0, color='#FFFFFF', lw=0.6, alpha=0.35)
ax2.set_title('Lasso — Coefficient Path  (features zeroed!)')
ax2.set_xlabel('log₁₀(α)'); ax2.set_ylabel('Coefficient')
ax2.legend(fontsize=6.5, loc='upper right', framealpha=0.12, ncol=2)
ax2.grid(True, alpha=0.35)
ax2.text(0.02, 0.05, 'Lines HIT zero → automatic feature selection', transform=ax2.transAxes,
         fontsize=7.5, color=CORAL, style='italic')

# ─── Plot 3: Coefficient bar comparison ────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(feature_names)); w = 0.26
ax3.bar(x - w, ols.coef_,   width=w, color=GREEN,  alpha=0.85, label='OLS',   zorder=3)
ax3.bar(x,     ridge.coef_, width=w, color=BLUE,   alpha=0.85, label='Ridge', zorder=3)
ax3.bar(x + w, lasso.coef_, width=w, color=CORAL,  alpha=0.85, label='Lasso', zorder=3)
ax3.axhline(0, color='#FFFFFF', lw=0.5, alpha=0.3)
ax3.set_xticks(x)
ax3.set_xticklabels(feature_names, rotation=38, ha='right', fontsize=8)
ax3.set_title('Coefficients: OLS vs Ridge vs Lasso')
ax3.set_ylabel('Coefficient value')
ax3.legend(fontsize=8, framealpha=0.15)
ax3.grid(True, axis='y', alpha=0.35)

# ─── Plot 4: Predicted vs Actual — Ridge ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ypr = ridge.predict(X_test)
ax4.scatter(y_test, ypr, alpha=0.25, s=7, color=BLUE, edgecolors='none')
mn, mx = y_test.min(), y_test.max()
ax4.plot([mn, mx], [mn, mx], color=AMBER, lw=1.5, ls='--', label='Perfect fit')
ax4.set_title(f'Ridge  Predicted vs Actual  R²={ridge_r2:.3f}')
ax4.set_xlabel('Actual value'); ax4.set_ylabel('Predicted')
ax4.legend(fontsize=8, framealpha=0.15); ax4.grid(True, alpha=0.3)

# ─── Plot 5: Predicted vs Actual — Lasso ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ypl = lasso.predict(X_test)
ax5.scatter(y_test, ypl, alpha=0.25, s=7, color=CORAL, edgecolors='none')
ax5.plot([mn, mx], [mn, mx], color=AMBER, lw=1.5, ls='--', label='Perfect fit')
ax5.set_title(f'Lasso  Predicted vs Actual  R²={lasso_r2:.3f}')
ax5.set_xlabel('Actual value'); ax5.set_ylabel('Predicted')
ax5.legend(fontsize=8, framealpha=0.15); ax5.grid(True, alpha=0.3)

# ─── Plot 6: Residual distribution ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
bins = np.linspace(-2.5, 2.5, 45)
ax6.hist(y_test - ypr, bins=bins, color=BLUE,  alpha=0.6, density=True,
         label=f'Ridge  RMSE={ridge_rmse:.3f}')
ax6.hist(y_test - ypl, bins=bins, color=CORAL, alpha=0.6, density=True,
         label=f'Lasso  RMSE={lasso_rmse:.3f}')
ax6.axvline(0, color='#FFFFFF', lw=0.8, alpha=0.4)
ax6.set_title('Residual Distribution')
ax6.set_xlabel('Actual − Predicted'); ax6.set_ylabel('Density')
ax6.legend(fontsize=8, framealpha=0.15); ax6.grid(True, alpha=0.3)

# ─── Plot 7: R² vs alpha — Ridge ─────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ridge_r2s = [r2_score(y_test, Ridge(alpha=a).fit(X_train,y_train).predict(X_test)) for a in alphas]
ax7.plot(np.log10(alphas), ridge_r2s, color=BLUE, lw=2)
ax7.fill_between(np.log10(alphas), ridge_r2s, min(ridge_r2s), alpha=0.15, color=BLUE)
ax7.axvline(np.log10(10), color=AMBER, lw=1.3, ls='--', alpha=0.8, label='α=10 chosen')
ax7.set_title('Ridge  R² vs  α')
ax7.set_xlabel('log₁₀(α)'); ax7.set_ylabel('R²')
ax7.legend(fontsize=8, framealpha=0.15); ax7.grid(True, alpha=0.3)

# ─── Plot 8: R² vs alpha — Lasso ─────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
lasso_r2s = [r2_score(y_test, Lasso(alpha=a,max_iter=10000).fit(X_train,y_train).predict(X_test)) for a in alphas]
ax8.plot(np.log10(alphas), lasso_r2s, color=CORAL, lw=2)
ax8.fill_between(np.log10(alphas), lasso_r2s, min(lasso_r2s), alpha=0.15, color=CORAL)
ax8.axvline(np.log10(0.05), color=AMBER, lw=1.3, ls='--', alpha=0.8, label='α=0.05 chosen')
ax8.set_title('Lasso  R² vs  α')
ax8.set_xlabel('log₁₀(α)'); ax8.set_ylabel('R²')
ax8.legend(fontsize=8, framealpha=0.15); ax8.grid(True, alpha=0.3)

# ─── Plot 9: Lasso feature selection (non-zero coefs) ────────────────────
ax9 = fig.add_subplot(gs[2, 2])
nonzero = np.array([(c != 0).sum() for c in lasso_coefs])
ax9.step(np.log10(alphas), nonzero, color=CORAL, lw=2.2, where='post')
ax9.fill_between(np.log10(alphas), nonzero, 0, step='post', alpha=0.18, color=CORAL)
ax9.axvline(np.log10(0.05), color=AMBER, lw=1.3, ls='--', alpha=0.8, label='α=0.05 chosen')
ax9.set_title('Lasso  Active Features  vs  α')
ax9.set_xlabel('log₁₀(α)'); ax9.set_ylabel('# non-zero coefs')
ax9.set_ylim(-0.3, len(feature_names) + 1)
ax9.legend(fontsize=8, framealpha=0.15); ax9.grid(True, alpha=0.3)
ax9.text(0.5, 0.55, '← Ridge always keeps all features', transform=ax9.transAxes,
         fontsize=7.5, color=BLUE, style='italic', ha='center')

plt.savefig('/mnt/user-data/outputs/ridge_vs_lasso.png', dpi=150, bbox_inches='tight',
            facecolor='#0F1117')
print(f"Done  OLS R²={ols_r2:.3f}  Ridge R²={ridge_r2:.3f}  Lasso R²={lasso_r2:.3f}")
