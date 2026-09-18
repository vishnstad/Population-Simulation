import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out_dir = r'd:\Documents\College_purpose\Final Year\Docs\presentations\First_Review\figures'
os.makedirs(out_dir, exist_ok=True)

html_path = r'd:\Documents\College_purpose\Final Year\results for 1st review\results for 1st review\popsim\runs\report\b17_results.html'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Save Standalone SVGs with embedded self-contained styles
svg_style = '''<style>
:root { --bg:#ffffff; --ink:#14130f; --ink2:#4e4c45; --ink3:#7c796f; --line:#e5e2d9; --accent:#2a78d6; --good:#008300; --bad:#e34948; }
text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.ch-title { font-size: 13px; font-weight: 600; fill: #4e4c45; }
.ch-label { font-size: 12.5px; fill: #4e4c45; }
.ch-value { font-size: 12px; font-weight: 500; fill: #14130f; font-family: monospace; }
.ch-tick { font-size: 11px; fill: #7c796f; font-family: monospace; }
.ch-axis { font-size: 12px; fill: #7c796f; }
.ch-legend { font-size: 12px; fill: #4e4c45; }
.ch-rule { stroke: #e34948; stroke-dasharray: 3 3; stroke-width: 1.5; }
.ch-rule-label { font-size: 11px; fill: #e34948; font-weight: 500; }
.ch-bar { fill: #7c796f; opacity: 0.45; }
.ch-bar-hi { fill: #2a78d6; opacity: 1; }
.ch-plot { fill: #ffffff; stroke: #e5e2d9; }
.ch-dot-good { fill: #008300; }
.ch-dot-bad { fill: #e34948; }
</style>'''

svgs = re.findall(r'<svg[^>]*>.*?</svg>', html, re.DOTALL)
names = ['01_macro_w1_all_arms.svg', '02_capability_curve.svg', '03_per_item_scatter.svg']
for name, raw_svg in zip(names, svgs):
    clean_svg = raw_svg.replace('style="max-width', 'xmlns="http://www.w3.org/2000/svg" style="background:#ffffff;max-width')
    clean_svg = clean_svg.replace('>', '>' + svg_style, 1)
    with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as f:
        f.write(clean_svg)
print('Saved 3 standalone SVG files!')

# 2. Render High-Res PNG & PDF for Chart 1: All Arms Macro W1
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

arms = [
    ('B3 (supervised skyline)', 0.0282, '#008300'),
    ('system (14B headline)', 0.0602, '#2a78d6'),
    ('system (topic-disjoint)', 0.0606, '#2a78d6'),
    ('system (s=1 unfitted)', 0.0609, '#2a78d6'),
    ('system (per-cluster scale)', 0.0612, '#2a78d6'),
    ('system (no subspace proj.)', 0.0626, '#2a78d6'),
    ('B0a (national marginal)', 0.0806, '#e34948'),
    ('B0a (cluster marginal)', 0.0807, '#e34948'),
    ('B0b (LLM-predicted marginal)', 0.1787, '#7c796f'),
    ('B4 (uncalibrated LLM)', 0.1926, '#7c796f'),
    ('B1 (nearest-anchor)', 0.2343, '#7c796f')
]
arms.reverse() # so best arm is at the top
labels = [a[0] for a in arms]
vals = [a[1] for a in arms]
colors = [a[2] for a in arms]

y_pos = np.arange(len(arms))
bars = ax.barh(y_pos, vals, color=colors, height=0.65, alpha=0.88, edgecolor='black', linewidth=0.6)

# Value annotations
for bar, val in zip(bars, vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
            va='center', ha='left', fontsize=9, fontweight='bold', color='#14130f')

# Noise floor dashed line
ax.axvline(0.0273, color='#e34948', linestyle='--', linewidth=1.5, label='Noise Floor (0.0273)')
ax.text(0.0273, -0.9, 'Noise floor (0.0273)', color='#e34948', ha='center', va='top', fontsize=8.5, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel('Macro $W_1$ Error (lower is better)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Macro $W_1$ Across All Arms (39 Scored Targets)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(0, 0.265)
ax.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '01_macro_w1_all_arms.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '01_macro_w1_all_arms.pdf'))
plt.close(fig)
print('Saved Chart 1 PNG & PDF!')

# 3. Render High-Res PNG & PDF for Chart 2: Capability Curve
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
models = ['Ministral 3B', 'Ministral 8B', 'Ministral 14B']
improvements = [-15.0, -21.4, -25.3] # % vs B0a
w1_scores = [0.0686, 0.0634, 0.0602]

x_pos = np.arange(len(models))
bars = ax.bar(x_pos, [-imp for imp in improvements], color=['#6baed6', '#3182bd', '#08519c'], width=0.48, edgecolor='black', linewidth=0.6)

for bar, imp, w1 in zip(bars, improvements, w1_scores):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.6, f'{imp}%\n($W_1={w1:.4f}$)', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10.5, fontweight='bold')
ax.set_ylabel('Error Reduction vs Baseline $B_{0a}$ (%)', fontsize=11, fontweight='bold')
ax.set_title('Capability Curve: Monotone Scaling Across Model Tiers', fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 32)
ax.axhline(20.0, color='#e34948', linestyle='--', linewidth=1.5, label='Pre-registered -20% Threshold')
ax.legend(loc='upper left', frameon=True, fontsize=9.5)
ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '02_capability_curve.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '02_capability_curve.pdf'))
plt.close(fig)
print('Saved Chart 2 PNG & PDF!')

# 4. Render High-Res PNG & PDF for Chart 3: Scatter Plot
items_data = re.findall(r'<title>([^:]+):\s*baseline\s*([0-9.]+),\s*system\s*([0-9.]+)</title>', html)
print(f'Parsed {len(items_data)} items from HTML for scatter plot')

b0a_vals = [float(item[1]) for item in items_data]
sys_vals = [float(item[2]) for item in items_data]
names_list = [item[0] for item in items_data]

fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=300)
ax.plot([0, 0.17], [0, 0.17], color='gray', linestyle='--', linewidth=1.2, label='$y = x$ (Parity Line)')

good_x, good_y, good_names = [], [], []
bad_x, bad_y, bad_names = [], [], []

for x, y, n in zip(b0a_vals, sys_vals, names_list):
    if y <= x:
        good_x.append(x)
        good_y.append(y)
        good_names.append(n)
    else:
        bad_x.append(x)
        bad_y.append(y)
        bad_names.append(n)

ax.scatter(good_x, good_y, color='#008300', s=48, alpha=0.88, edgecolors='black', linewidth=0.5, label=f'System Better ({len(good_x)} items)')
ax.scatter(bad_x, bad_y, color='#e34948', s=58, alpha=0.92, edgecolors='black', linewidth=0.7, label=f'Baseline Better ({len(bad_x)} items)')

# Annotate the 3 baseline-better items
for x, y, n in zip(bad_x, bad_y, bad_names):
    ax.annotate(n, (x, y), textcoords='offset points', xytext=(7, -4), fontsize=9, fontweight='bold', color='#e34948')

# Annotate notable wins
for win in ['xmovie', 'pray', 'colath', 'homosex']:
    if win in good_names:
        idx = good_names.index(win)
        ax.annotate(win, (good_x[idx], good_y[idx]), textcoords='offset points', xytext=(6, -3), fontsize=8, color='#008300')

ax.set_xlim(0, 0.17)
ax.set_ylim(0, 0.17)
ax.set_aspect('equal')
ax.set_xlabel('Baseline $B_{0a}$ $W_1$ (National Marginal)', fontsize=11, fontweight='bold')
ax.set_ylabel('System $W_1$ (Calibrated Subgroup Prediction)', fontsize=11, fontweight='bold')
ax.set_title('Item-by-Item Performance: Calibrated System vs Baseline', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='lower right', frameon=True, fontsize=9.5)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '03_per_item_scatter.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '03_per_item_scatter.pdf'))
plt.close(fig)
print('Saved Chart 3 PNG & PDF!')

# 5. Render High-Res System Architecture Diagram
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

box_props = dict(boxstyle='round,pad=0.5', facecolor='#e8f1fb', edgecolor='#1a4971', linewidth=1.5)
core_props = dict(boxstyle='round,pad=0.5', facecolor='#fee8d6', edgecolor='#d95f02', linewidth=1.8)
obs_props = dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#666666', linewidth=1.2)
diam_props = dict(boxstyle='square,pad=0.4', facecolor='#ffffff', edgecolor='#1a4971', linewidth=1.5)

# Row 1: Pipeline
ax.text(1.2, 3.8, 'M1\nIngest &\nHarmonize', ha='center', va='center', bbox=box_props, fontsize=9.5, fontweight='bold')
ax.text(3.4, 3.8, 'M2\nCluster Tree\n(K=56)', ha='center', va='center', bbox=box_props, fontsize=9.5, fontweight='bold')
ax.text(5.6, 3.8, 'M3\nStat Card\nBuilder', ha='center', va='center', bbox=box_props, fontsize=9.5, fontweight='bold')
ax.text(7.8, 3.8, 'M7\nOracle\nRouter', ha='center', va='center', bbox=diam_props, fontsize=9.5, fontweight='bold')
ax.text(10.2, 3.8, 'Observed\nCross-Tab\n(Zero Cost)', ha='center', va='center', bbox=obs_props, fontsize=9.5)

# Row 2: LLM & Calibration
ax.text(5.6, 1.5, 'M4\nDistributional\nElicitation (LLM)', ha='center', va='center', bbox=core_props, fontsize=9.5, fontweight='bold')
ax.text(7.8, 1.5, 'M5\nCalibration Layer\n(Isotonic + $P_r$)', ha='center', va='center', bbox=core_props, fontsize=9.5, fontweight='bold')
ax.text(10.0, 1.5, 'M6 / M9 / M10\nCensus Raking &\nEvaluation App', ha='center', va='center', bbox=box_props, fontsize=9.5, fontweight='bold')

# Arrows
arr_kw = dict(arrowstyle='->', lw=1.8, color='#1a4971')
ax.annotate('', xy=(2.3, 3.8), xytext=(1.9, 3.8), arrowprops=arr_kw)
ax.annotate('', xy=(4.5, 3.8), xytext=(4.1, 3.8), arrowprops=arr_kw)
ax.annotate('', xy=(6.7, 3.8), xytext=(6.3, 3.8), arrowprops=arr_kw)
ax.annotate('', xy=(8.9, 3.8), xytext=(8.5, 3.8), arrowprops=arr_kw)
ax.text(8.7, 4.05, 'observed', fontsize=8.5, color='#666666')

# Routing unseen
ax.annotate('', xy=(5.6, 2.3), xytext=(7.8, 3.2), arrowprops=dict(arrowstyle='->', lw=1.8, color='#d95f02', connectionstyle='arc3,rad=-0.15'))
ax.text(6.8, 2.9, 'unseen question', fontsize=8.5, fontweight='bold', color='#d95f02')

ax.annotate('', xy=(6.8, 1.5), xytext=(6.5, 1.5), arrowprops=arr_kw)
ax.annotate('', xy=(9.0, 1.5), xytext=(8.7, 1.5), arrowprops=arr_kw)

plt.title('B17 System Architecture: Conditional Distribution Transfer Pipeline', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '04_system_architecture.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '04_system_architecture.pdf'))
plt.close(fig)
print('Saved Architecture PNG & PDF!')
