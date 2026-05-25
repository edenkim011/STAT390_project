"""
CV stability visualization for the Exp_36 final GBR model.
Runs the complete 5-fold CV loop (tuned GBR + within-fold TF-IDF) on
data/working_train.csv and plots fold-level RMSE as a side-by-side
box-and-whisker + per-fold bar chart.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from prepare import evaluate_model


# ── locked Exp_36 configuration ────────────────────────────────────────────
BASE_FEATURES = [
    'Age', 'Fee', 'Type', 'Gender', 'Health', 'Vaccinated',
    'Dewormed', 'Sterilized', 'PhotoAmt', 'Quantity',
    'MaturitySize', 'FurLength', 'Breed1', 'Color1',
    'Breed2', 'Color2', 'Color3', 'State', 'VideoAmt',
]
SENTIMENT_FEATURES = [
    'doc_score', 'doc_magnitude',
    'sent_count', 'sent_score_mean', 'sent_mag_mean',
    'sent_score_std', 'sent_score_max', 'sent_score_min', 'sent_mag_max',
    'entity_count', 'top_salience',
]
VISUAL_FEATURES = [
    'top_color_score', 'top_color_pixel_frac',
    'color_count', 'top3_pixel_frac_sum',
    'crop_confidence', 'crop_importance',
    'top_label_score', 'label_count', 'label_score_mean',
]
N_TFIDF = 15

GBR_PARAMS = dict(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, subsample=0.8, min_samples_leaf=10,
    random_state=42,
)
REPORTED_MEAN = 1.0643   # Exp_36 result from research log


# ── data loaders ───────────────────────────────────────────────────────────
def load_sentiment(pet_ids, sentiment_dir='data/train_sentiment'):
    records = []
    for pid in pet_ids:
        path = os.path.join(sentiment_dir, f'{pid}.json')
        if not os.path.exists(path):
            records.append({'PetID': pid})
            continue
        with open(path) as f:
            d = json.load(f)
        sentences = d.get('sentences', [])
        entities  = d.get('entities', [])
        doc       = d.get('documentSentiment', {})
        scores = [s['sentiment']['score']     for s in sentences]
        mags   = [s['sentiment']['magnitude'] for s in sentences]
        records.append({
            'PetID':           pid,
            'doc_score':       doc.get('score', 0),
            'doc_magnitude':   doc.get('magnitude', 0),
            'sent_count':      len(sentences),
            'sent_score_mean': np.mean(scores) if scores else 0,
            'sent_score_max':  np.max(scores)  if scores else 0,
            'sent_score_min':  np.min(scores)  if scores else 0,
            'sent_score_std':  np.std(scores)  if scores else 0,
            'sent_mag_mean':   np.mean(mags)   if mags   else 0,
            'sent_mag_max':    np.max(mags)    if mags   else 0,
            'entity_count':    len(entities),
            'top_salience':    entities[0]['salience'] if entities else 0,
        })
    return pd.DataFrame(records).fillna(0)


def load_metadata(pet_ids, metadata_dir='data/train_metadata', agg='mean'):
    pet_photos = defaultdict(list)
    for fname in os.listdir(metadata_dir):
        if fname.endswith('.json'):
            pid = fname.rsplit('-', 1)[0]
            pet_photos[pid].append(fname)
    records = []
    for pid in pet_ids:
        photos = sorted(pet_photos.get(pid, []))
        if not photos:
            records.append({'PetID': pid})
            continue
        per_photo = []
        for fname in photos:
            with open(os.path.join(metadata_dir, fname)) as f:
                d = json.load(f)
            colors = d.get('imagePropertiesAnnotation', {}) \
                      .get('dominantColors', {}).get('colors', [])
            crops  = d.get('cropHintsAnnotation', {}).get('cropHints', [])
            labels = d.get('labelAnnotations', [])
            top3_pf      = sum(c['pixelFraction'] for c in colors[:3]) if colors else 0
            label_scores = [l['score'] for l in labels]
            per_photo.append({
                'top_color_score':      colors[0]['score']           if colors else 0,
                'top_color_pixel_frac': colors[0]['pixelFraction']   if colors else 0,
                'color_count':          len(colors),
                'top3_pixel_frac_sum':  top3_pf,
                'crop_confidence':      crops[0].get('confidence', 0)        if crops else 0,
                'crop_importance':      crops[0].get('importanceFraction', 0) if crops else 0,
                'top_label_score':      labels[0]['score']           if labels else 0,
                'label_count':          len(labels),
                'label_score_mean':     np.mean(label_scores)        if label_scores else 0,
            })
        df_p = pd.DataFrame(per_photo)
        row = df_p.max().to_dict() if agg == 'max' else df_p.mean().to_dict()
        row['PetID'] = pid
        records.append(row)
    return pd.DataFrame(records).fillna(0)


# ── cv loop ────────────────────────────────────────────────────────────────
def run_cv(train_df):
    all_features = BASE_FEATURES + SENTIMENT_FEATURES + VISUAL_FEATURES
    X_base = train_df[all_features].values
    y      = train_df['AdoptionSpeed'].values
    descs  = train_df['Description'].apply(
        lambda x: x if isinstance(x, str) else ''
    ).values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_base), 1):
        X_train = X_base[train_idx].copy()
        X_val   = X_base[val_idx].copy()
        y_train = y[train_idx]
        y_val   = y[val_idx]

        # scale Age and Fee (first two columns) — fit on train fold only
        scaler = StandardScaler()
        X_train[:, :2] = scaler.fit_transform(X_train[:, :2])
        X_val[:, :2]   = scaler.transform(X_val[:, :2])

        # within-fold TF-IDF — fit on train descriptions only
        tfidf = TfidfVectorizer(
            max_features=N_TFIDF, stop_words='english', sublinear_tf=True
        )
        tfidf_train = tfidf.fit_transform(descs[train_idx]).toarray()
        tfidf_val   = tfidf.transform(descs[val_idx]).toarray()

        X_train = np.column_stack([X_train, tfidf_train])
        X_val   = np.column_stack([X_val,   tfidf_val])

        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X_train, y_train)

        rmse = evaluate_model(y_val, model.predict(X_val))
        fold_rmses.append(rmse)
        print(f"  Fold {fold}  RMSE: {rmse:.4f}")

    return fold_rmses


# ── plot ───────────────────────────────────────────────────────────────────
BLUE      = '#4C72B0'   # bar / box fill
BLUE_DARK = '#2d4f85'   # box median line, point edge
BLUE_LITE = '#d0ddf5'   # box fill (lighter)
ORANGE    = '#DD8452'   # mean reference line


def apply_spine_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='both', length=0)


def plot_stability(fold_rmses, output='cv_stability.png'):
    computed_mean = np.mean(fold_rmses)
    computed_std  = np.std(fold_rmses)
    fold_labels   = [f'Fold {i}' for i in range(1, 6)]

    # y-axis window: span the data with equal padding on both sides
    y_lo = min(fold_rmses) - 0.006
    y_hi = max(fold_rmses) + 0.006

    fig, (ax_box, ax_bar) = plt.subplots(
        1, 2,
        figsize=(13, 6),
        gridspec_kw={'width_ratios': [1, 1.6]},
    )
    fig.suptitle(
        'Exp_36 GBR — 5-Fold CV Stability (tuned GBR + within-fold TF-IDF)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    # ── left: box & whisker ────────────────────────────────────────────────
    bp = ax_box.boxplot(
        fold_rmses,
        widths=0.45,
        patch_artist=True,
        whis=[0, 100],           # whiskers reach exact min/max
        showfliers=False,
        medianprops=dict(color=BLUE_DARK, linewidth=2.2),
        whiskerprops=dict(color=BLUE, linewidth=1.4, linestyle='--'),
        capprops=dict(color=BLUE, linewidth=1.6),
        boxprops=dict(facecolor=BLUE_LITE, edgecolor=BLUE, linewidth=1.5),
    )

    # overlay individual fold points with slight x-jitter
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.08, 0.08, size=len(fold_rmses))
    ax_box.scatter(
        np.ones(len(fold_rmses)) + jitter, fold_rmses,
        color=BLUE, edgecolors=BLUE_DARK, s=52, zorder=5, linewidths=0.8,
        label='Individual fold',
    )

    # annotate median
    median_val = np.median(fold_rmses)
    ax_box.text(
        1.32, median_val,
        f'median\n{median_val:.4f}',
        va='center', ha='left', fontsize=8.5,
        color=BLUE_DARK, fontweight='bold',
    )

    # mean reference line
    ax_box.axhline(computed_mean, color=ORANGE, linewidth=1.6,
                   linestyle='--', zorder=3)
    ax_box.text(
        0.62, computed_mean + (y_hi - y_lo) * 0.015,
        f'mean {computed_mean:.4f}',
        fontsize=8, color=ORANGE, ha='center',
    )

    ax_box.set_xticks([1])
    ax_box.set_xticklabels(['5-Fold\nDistribution'], fontsize=9.5)
    ax_box.set_ylabel('CV RMSE', fontsize=10, labelpad=8)
    ax_box.set_ylim(y_lo, y_hi)
    ax_box.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax_box.set_title('Distribution & Spread', fontsize=10.5, pad=8)
    ax_box.grid(axis='y', linestyle='--', linewidth=0.5, color='#e8e8e8', zorder=0)
    ax_box.set_axisbelow(True)
    apply_spine_style(ax_box)

    # ── right: per-fold bar chart ──────────────────────────────────────────
    x = np.arange(5)
    # shade bars by how close they are to the mean
    bar_colors = [
        BLUE if abs(v - computed_mean) <= computed_std else '#8aabdb'
        for v in fold_rmses
    ]
    bars = ax_bar.bar(x, fold_rmses, color=bar_colors, width=0.52,
                      edgecolor='none', zorder=3)

    # value labels on top of each bar
    label_offset = (y_hi - y_lo) * 0.006
    for rect, val in zip(bars, fold_rmses):
        ax_bar.text(
            rect.get_x() + rect.get_width() / 2,
            val + label_offset,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, color='#333333',
        )

    # mean reference line with label
    ax_bar.axhline(
        REPORTED_MEAN, color=ORANGE, linewidth=2, linestyle='--', zorder=4,
        label=f'Mean CV RMSE = {REPORTED_MEAN:.4f}  (Exp_36 reported)',
    )
    # computed mean annotation if it differs from reported
    if abs(computed_mean - REPORTED_MEAN) >= 0.0001:
        ax_bar.axhline(
            computed_mean, color='#888888', linewidth=1.2, linestyle=':',
            zorder=4, label=f'Computed mean = {computed_mean:.4f}',
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(fold_labels, fontsize=9.5)
    ax_bar.set_ylabel('CV RMSE', fontsize=10, labelpad=8)
    ax_bar.set_ylim(y_lo, y_hi)
    ax_bar.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax_bar.set_title('Per-Fold RMSE Tracking', fontsize=10.5, pad=8)
    ax_bar.grid(axis='y', linestyle='--', linewidth=0.5, color='#e8e8e8', zorder=0)
    ax_bar.set_axisbelow(True)
    apply_spine_style(ax_bar)

    leg = ax_bar.legend(
        loc='upper right', fontsize=8.5,
        framealpha=0.92, edgecolor='#cccccc',
    )

    # ── shared footer stats ────────────────────────────────────────────────
    spread = max(fold_rmses) - min(fold_rmses)
    fig.text(
        0.5, -0.04,
        f'n=5 folds  ·  mean={computed_mean:.4f}  ·  std={computed_std:.4f}  '
        f'·  min={min(fold_rmses):.4f}  ·  max={max(fold_rmses):.4f}  '
        f'·  range={spread:.4f}',
        ha='center', fontsize=8.5, color='#666666',
    )

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {output}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading data/working_train.csv …")
    train_df = pd.read_csv('data/working_train.csv')

    print("Loading sentiment features …")
    sent_df  = load_sentiment(train_df['PetID'].tolist())
    train_df = train_df.merge(sent_df, on='PetID', how='left').fillna(0)

    print("Loading visual metadata features (mean aggregation) …")
    meta_df  = load_metadata(train_df['PetID'].tolist(), agg='mean')
    train_df = train_df.merge(meta_df, on='PetID', how='left').fillna(0)

    print(f"\nRunning 5-fold CV  (GBR n={GBR_PARAMS['n_estimators']}, "
          f"lr={GBR_PARAMS['learning_rate']}, depth={GBR_PARAMS['max_depth']}, "
          f"subsample={GBR_PARAMS['subsample']}, leaf={GBR_PARAMS['min_samples_leaf']}) …")
    fold_rmses = run_cv(train_df)

    mean_rmse = np.mean(fold_rmses)
    std_rmse  = np.std(fold_rmses)
    print(f"\n5-fold CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Reported Exp_36:  {REPORTED_MEAN:.4f}")
    print(f"Delta from log:   {abs(mean_rmse - REPORTED_MEAN):.4f}")

    print("\nGenerating cv_stability.png …")
    plot_stability(fold_rmses, output='cv_stability.png')


if __name__ == '__main__':
    main()
