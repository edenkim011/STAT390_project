import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

KEEP_COLOR    = '#27ae60'
DISCARD_COLOR = '#e74c3c'
LINE_COLOR    = '#bdc3c7'
BEST_CV_RMSE  = 1.0643

def point_color(decision):
    return KEEP_COLOR if decision == 'KEEP' else DISCARD_COLOR

def create_trajectory_plot():
    df = pd.read_csv('results.tsv', sep='\t')
    df['Metric_RMSE'] = pd.to_numeric(df['Metric_RMSE'], errors='coerce')

    def exp_num(eid):
        if eid == 'Baseline_01':
            return 0
        if eid.startswith('Exp_'):
            return int(eid.split('_')[1])
        return -1

    df['_n'] = df['Experiment_ID'].apply(exp_num)
    era1 = df[df['_n'] <= 22].copy().reset_index(drop=True)
    era2 = df[df['_n'] >= 23].copy().reset_index(drop=True)
    era1['x'] = range(len(era1))
    era2['x'] = range(len(era2))

    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor('#ffffff')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── ERA 1: Training RMSE ────────────────────────────────────────────────
    ax1.set_facecolor('#fdfaf6')
    ax1.plot(era1['x'], era1['Metric_RMSE'],
             linestyle='-', color=LINE_COLOR, linewidth=1.8, zorder=1)

    for _, row in era1.iterrows():
        ax1.scatter(row['x'], row['Metric_RMSE'],
                    color=point_color(row['Decision']),
                    s=95, edgecolors='white', linewidths=0.9, zorder=3)

    # Label non-zero points; label only first 0.0000 to avoid clutter
    seen_zero = False
    for _, row in era1.iterrows():
        rmse = row['Metric_RMSE']
        if rmse == 0.0:
            if not seen_zero:
                ax1.annotate('0.0000\n(Exp_18–22\noverfit)',
                             xy=(row['x'], 0),
                             xytext=(row['x'] + 1.5, 0.12),
                             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
                             fontsize=8, color='#c0392b', fontweight='bold',
                             ha='left')
                seen_zero = True
        else:
            offset = 0.038 if rmse > 0.15 else 0.025
            ax1.text(row['x'], rmse + offset,
                     f'{rmse:.4f}',
                     ha='center', va='bottom', fontsize=7.8,
                     fontweight='bold', color='#2c3e50')

    # Overfit floor band
    ax1.axhspan(-0.02, 0.045, alpha=0.10, color='#e74c3c', zorder=0)
    ax1.axhline(y=0, color='#e74c3c', linestyle='--', alpha=0.55, linewidth=1.2)

    # Era label banner
    ax1.text(0.5, 0.97,
             'ERA 1 — Training RMSE\nModel memorizes training set → RMSE collapses to 0.0000',
             transform=ax1.transAxes, ha='center', va='top',
             fontsize=10, color='#7b241c', style='italic',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#fadbd8', alpha=0.75))

    ax1.set_xticks(era1['x'])
    ax1.set_xticklabels(era1['Experiment_ID'], rotation=48, ha='right', fontsize=8.2)
    ax1.set_title('Baseline → Exp_22', fontsize=12, fontweight='bold',
                  color='#2c3e50', pad=6)
    ax1.set_ylabel('RMSE (training)', fontsize=12, color='#2c3e50')
    ax1.set_xlabel('Experiment', fontsize=11, color='#2c3e50')
    ax1.set_ylim(-0.06, 1.38)
    ax1.set_xlim(-0.6, len(era1) - 0.4)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.45)
    ax1.spines[['top', 'right']].set_visible(False)

    # ── ERA 2: 5-Fold CV RMSE ───────────────────────────────────────────────
    ax2.set_facecolor('#f5fbf5')
    ax2.plot(era2['x'], era2['Metric_RMSE'],
             linestyle='-', color=LINE_COLOR, linewidth=1.8, zorder=1)

    for _, row in era2.iterrows():
        ax2.scatter(row['x'], row['Metric_RMSE'],
                    color=point_color(row['Decision']),
                    s=95, edgecolors='white', linewidths=0.9, zorder=3)
        ax2.text(row['x'], row['Metric_RMSE'] + 0.0007,
                 f'{row["Metric_RMSE"]:.4f}',
                 ha='center', va='bottom', fontsize=7.8,
                 fontweight='bold', color='#2c3e50')

    # Best CV floor line
    ax2.axhline(y=BEST_CV_RMSE, color=KEEP_COLOR, linestyle='--',
                alpha=0.75, linewidth=1.6)
    ax2.text(len(era2) - 0.3, BEST_CV_RMSE - 0.0013,
             f'Best: {BEST_CV_RMSE:.4f}  (Exp_36)',
             ha='right', va='top', fontsize=9,
             color='#1e8449', fontweight='bold')

    # Era label banner
    ax2.text(0.5, 0.97,
             'ERA 2 — 5-Fold Cross-Validation\nTrue out-of-sample floor locked at ~1.0643',
             transform=ax2.transAxes, ha='center', va='top',
             fontsize=10, color='#1a5e34', style='italic',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#d5f5e3', alpha=0.75))

    ax2.set_xticks(era2['x'])
    ax2.set_xticklabels(era2['Experiment_ID'], rotation=48, ha='right', fontsize=8.2)
    ax2.set_title('Exp_23 → Exp_37', fontsize=12, fontweight='bold',
                  color='#2c3e50', pad=6)
    ax2.set_ylabel('CV RMSE (5-fold)', fontsize=12, color='#2c3e50')
    ax2.set_xlabel('Experiment', fontsize=11, color='#2c3e50')

    y_lo = era2['Metric_RMSE'].min() - 0.004
    y_hi = era2['Metric_RMSE'].max() + 0.008
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlim(-0.6, len(era2) - 0.4)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.45)
    ax2.spines[['top', 'right']].set_visible(False)

    # ── Shared legend & super-title ─────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='KEEP (improvement kept)',
               markerfacecolor=KEEP_COLOR, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='DISCARD (regression rolled back)',
               markerfacecolor=DISCARD_COLOR, markersize=10),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=2, fontsize=11, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle('Pet Adoption Speed — Full Experiment Trajectory',
                 fontsize=17, fontweight='bold', color='#1a252f', y=1.065)

    plt.savefig('trajectory_plot.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    print('Successfully generated: trajectory_plot.png')

if __name__ == '__main__':
    create_trajectory_plot()
