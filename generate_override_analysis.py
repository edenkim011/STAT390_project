"""
Override analysis: can linguistic framing override the physical age adoption bias?

For each pet: compute a composite TF-IDF text score (sum of 15 within-sample
unigram weights) and a GBR-predicted AdoptionSpeed using the locked Exp_36 config.
Scatter x=Age, y=prediction, color=text intensity. Two linear trend lines compare
the age-speed trajectory for low- vs high-text-signal listings.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# ── locked Exp_36 feature definitions ─────────────────────────────────────
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

LOW_QUANTILE  = 0.25   # bottom 25 %  → "sparse listing text"
HIGH_QUANTILE = 0.75   # top    25 %  → "rich listing text"


# ── data loaders (identical to Exp_36 pipeline) ────────────────────────────
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


# ── pipeline ───────────────────────────────────────────────────────────────
def build_pipeline(train_df):
    """Return X (full feature matrix), text_scores, y, and fitted TF-IDF vocab."""
    all_features = BASE_FEATURES + SENTIMENT_FEATURES + VISUAL_FEATURES
    X_tab = train_df[all_features].values.copy()

    scaler = StandardScaler()
    X_tab[:, :2] = scaler.fit_transform(X_tab[:, :2])   # Age, Fee only

    descriptions = (
        train_df['Description']
        .apply(lambda x: x if isinstance(x, str) else '')
        .values
    )
    tfidf = TfidfVectorizer(
        max_features=N_TFIDF, stop_words='english', sublinear_tf=True
    )
    X_tfidf = tfidf.fit_transform(descriptions).toarray()

    # composite text score: per-row sum of all 15 TF-IDF weights
    text_scores = X_tfidf.sum(axis=1)

    X = np.column_stack([X_tab, X_tfidf])
    y = train_df['AdoptionSpeed'].values
    return X, text_scores, y, tfidf.get_feature_names_out()


def trend_line(x, y, age_grid):
    """Fit OLS and return (fitted y-values on age_grid, r-squared, p-value)."""
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return slope * age_grid + intercept, r ** 2, p, slope


# ── plot ───────────────────────────────────────────────────────────────────
def plot_override(age, y_pred, text_scores, tfidf_tokens, output='override_analysis.png'):
    rng = np.random.default_rng(0)
    age_jitter = age + rng.uniform(-0.4, 0.4, size=len(age))   # month-level jitter

    low_cut  = np.percentile(text_scores, LOW_QUANTILE  * 100)
    high_cut = np.percentile(text_scores, HIGH_QUANTILE * 100)
    low_mask  = text_scores <= low_cut
    high_mask = text_scores >= high_cut

    age_grid = np.linspace(np.percentile(age, 2), np.percentile(age, 98), 300)

    trend_lo, r2_lo, p_lo, slope_lo = trend_line(age[low_mask],  y_pred[low_mask],  age_grid)
    trend_hi, r2_hi, p_hi, slope_hi = trend_line(age[high_mask], y_pred[high_mask], age_grid)

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8))
    # main axes takes most of the width; right strip for colorbar annotation
    ax = fig.add_axes([0.08, 0.13, 0.78, 0.74])

    # ── scatter ────────────────────────────────────────────────────────────
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=text_scores.min(), vmax=text_scores.max())

    sc = ax.scatter(
        age_jitter, y_pred,
        c=text_scores, cmap=cmap, norm=norm,
        s=18, alpha=0.55, linewidths=0,
        zorder=2,
    )

    # ── trend lines ────────────────────────────────────────────────────────
    lo_color = '#1e0533'   # very dark purple  (viridis low end)
    hi_color = '#f0e442'   # golden yellow     (viridis high end)

    ax.plot(
        age_grid, trend_lo,
        color=lo_color, linewidth=2.8, linestyle='-',
        label=(
            f'Low text signal  (bottom {int(LOW_QUANTILE*100)}%)  '
            f'slope={slope_lo:+.4f}  R²={r2_lo:.3f}'
        ),
        zorder=5,
    )
    ax.plot(
        age_grid, trend_hi,
        color=hi_color, linewidth=2.8, linestyle='--',
        label=(
            f'High text signal (top {int((1-HIGH_QUANTILE)*100)}%)  '
            f'slope={slope_hi:+.4f}  R²={r2_hi:.3f}'
        ),
        zorder=5,
    )

    # gap annotation: vertical arrow at the median age showing override delta
    mid_age_idx = np.argmin(np.abs(age_grid - np.median(age)))
    delta = trend_lo[mid_age_idx] - trend_hi[mid_age_idx]
    if delta > 0.02:
        ax.annotate(
            '',
            xy=(age_grid[mid_age_idx], trend_hi[mid_age_idx]),
            xytext=(age_grid[mid_age_idx], trend_lo[mid_age_idx]),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.4),
            zorder=6,
        )
        ax.text(
            age_grid[mid_age_idx] + 0.8,
            (trend_lo[mid_age_idx] + trend_hi[mid_age_idx]) / 2,
            f'Δ = {delta:.3f}\nspeed units',
            fontsize=8, color='#444444', va='center',
        )

    # ── axes style ─────────────────────────────────────────────────────────
    ax.set_xlabel('Pet Age (months)', fontsize=11, labelpad=8)
    ax.set_ylabel(
        'Predicted AdoptionSpeed\n'
        '← faster adoption  |  slower adoption →',
        fontsize=11, labelpad=8,
    )
    ax.set_title(
        'Linguistic Override Analysis — Exp_36 GBR\n'
        'Does richer description vocabulary overcome the age-adoption penalty?',
        fontsize=13, fontweight='bold', pad=14,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='both', length=0, labelsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#e8e8e8', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    # y-axis: lock to AdoptionSpeed range with small padding
    ax.set_ylim(-0.15, 4.25)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(
        ['0\n(same week)', '1\n(≤7 days)', '2\n(≤30 days)', '3\n(≤90 days)', '4\n(≥100 days)'],
        fontsize=8,
    )

    # legend (inside plot, upper left)
    leg = ax.legend(
        loc='upper left', fontsize=8.5,
        framealpha=0.9, edgecolor='#cccccc',
        title='Trend line groups', title_fontsize=8.5,
    )

    # ── colorbar ───────────────────────────────────────────────────────────
    cax = fig.add_axes([0.88, 0.13, 0.025, 0.74])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(
        'Text Vocabulary Intensity\n(Composite TF-IDF Score)',
        fontsize=9.5, labelpad=10,
    )
    cbar.ax.tick_params(labelsize=8)
    # annotate colorbar poles
    cbar.ax.text(
        2.8, 0.02, 'sparse\n(few keywords)', transform=cbar.ax.transAxes,
        fontsize=7.5, color='#555555', va='bottom',
    )
    cbar.ax.text(
        2.8, 0.98, 'rich\n(many keywords)', transform=cbar.ax.transAxes,
        fontsize=7.5, color='#555555', va='top',
    )

    # ── top-token footnote ─────────────────────────────────────────────────
    token_str = '  ·  '.join(tfidf_tokens[:10])
    fig.text(
        0.08, 0.04,
        f'Top TF-IDF unigrams driving vocabulary score:  {token_str}  … (+{N_TFIDF - 10} more)',
        fontsize=7.5, color='#777777', style='italic',
    )

    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {output}")


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

    print("Building feature matrix (39 tabular + 15 TF-IDF) …")
    X, text_scores, y, tfidf_tokens = build_pipeline(train_df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Text score range: [{text_scores.min():.4f}, {text_scores.max():.4f}]  "
          f"median={np.median(text_scores):.4f}")

    print("Fitting Exp_36 GBR …")
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"  Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")

    age = train_df['Age'].values

    # summary statistics per text-signal group
    low_cut  = np.percentile(text_scores, LOW_QUANTILE  * 100)
    high_cut = np.percentile(text_scores, HIGH_QUANTILE * 100)
    low_mask  = text_scores <= low_cut
    high_mask = text_scores >= high_cut

    print(f"\nGroup summary:")
    print(f"  Low text signal  (score ≤ {low_cut:.4f}):  "
          f"n={low_mask.sum()}  mean_age={age[low_mask].mean():.1f}  "
          f"mean_pred={y_pred[low_mask].mean():.3f}")
    print(f"  High text signal (score ≥ {high_cut:.4f}):  "
          f"n={high_mask.sum()}  mean_age={age[high_mask].mean():.1f}  "
          f"mean_pred={y_pred[high_mask].mean():.3f}")

    print("\nGenerating override_analysis.png …")
    plot_override(age, y_pred, text_scores, tfidf_tokens, output='override_analysis.png')


if __name__ == '__main__':
    main()
