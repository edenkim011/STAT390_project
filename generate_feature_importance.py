"""
Feature importance visualization for Exp_36 optimal GBR configuration.
Single-pass fit on all training data to extract model.feature_importances_.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# ── feature group definitions ──────────────────────────────────────────────
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

# aggregated category labels
LABEL_SENTIMENT = 'Google NLP Sentiment Aggregates'
LABEL_VISUAL    = 'Google Vision Image Metadata'
LABEL_TFIDF     = 'TF-IDF Description Vocabulary'

# color palette
COLOR_BASE      = '#4C72B0'   # muted blue — individual tabular features
COLOR_SENTIMENT = '#DD8452'   # warm orange — NLP aggregate
COLOR_VISUAL    = '#55A868'   # green — Vision aggregate
COLOR_TFIDF     = '#8172B3'   # purple — TF-IDF aggregate


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


def build_features(train_df):
    features = BASE_FEATURES + SENTIMENT_FEATURES + VISUAL_FEATURES
    X_tab = train_df[features].values.copy()

    scaler = StandardScaler()
    X_tab[:, :2] = scaler.fit_transform(X_tab[:, :2])   # Age, Fee

    descriptions = train_df['Description'] \
        .apply(lambda x: x if isinstance(x, str) else '').values

    tfidf = TfidfVectorizer(max_features=N_TFIDF, stop_words='english', sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(descriptions).toarray()
    tfidf_tokens = tfidf.get_feature_names_out()

    X = np.column_stack([X_tab, X_tfidf])
    return X, features, tfidf_tokens


def fit_model(X, y):
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X, y)
    return model


def aggregate_importances(importances, feature_names, tfidf_tokens):
    """Return a dict of {display_label: importance_value} for the bar chart."""
    n_base      = len(BASE_FEATURES)
    n_sentiment = len(SENTIMENT_FEATURES)
    n_visual    = len(VISUAL_FEATURES)

    base_imp      = dict(zip(BASE_FEATURES, importances[:n_base]))
    sentiment_imp = importances[n_base : n_base + n_sentiment]
    visual_imp    = importances[n_base + n_sentiment : n_base + n_sentiment + n_visual]
    tfidf_imp     = importances[n_base + n_sentiment + n_visual:]

    # individual token breakdown (printed, not plotted separately)
    print("\nTF-IDF token importances:")
    for tok, imp in sorted(zip(tfidf_tokens, tfidf_imp), key=lambda x: -x[1]):
        print(f"  {tok:<20s}  {imp:.5f}")

    print("\nSentiment feature importances:")
    for feat, imp in zip(SENTIMENT_FEATURES, sentiment_imp):
        print(f"  {feat:<25s}  {imp:.5f}")

    print("\nVisual feature importances:")
    for feat, imp in zip(VISUAL_FEATURES, visual_imp):
        print(f"  {feat:<25s}  {imp:.5f}")

    result = {}
    result.update(base_imp)
    result[LABEL_SENTIMENT] = float(sentiment_imp.sum())
    result[LABEL_VISUAL]    = float(visual_imp.sum())
    result[LABEL_TFIDF]     = float(tfidf_imp.sum())
    return result


def bar_color(label):
    if label == LABEL_SENTIMENT:
        return COLOR_SENTIMENT
    if label == LABEL_VISUAL:
        return COLOR_VISUAL
    if label == LABEL_TFIDF:
        return COLOR_TFIDF
    return COLOR_BASE


def plot_importance(importance_dict, output_path='feature_importance.png'):
    # sort highest → lowest
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1])  # ascending for barh
    labels  = [item[0] for item in sorted_items]
    values  = [item[1] for item in sorted_items]
    colors  = [bar_color(lbl) for lbl in labels]

    n = len(labels)
    fig_height = max(8, n * 0.38)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    bars = ax.barh(range(n), values, color=colors, height=0.65, edgecolor='none')

    # numeric labels at end of each bar
    x_max = max(values)
    for i, (val, bar) in enumerate(zip(values, bars)):
        ax.text(
            val + x_max * 0.008, i,
            f'{val:.4f}',
            va='center', ha='left',
            fontsize=8.5, color='#333333', fontweight='normal',
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Feature Importance (mean decrease in impurity)', fontsize=10, labelpad=8)
    ax.set_title(
        'Exp_36 GBR — Feature Importance\n'
        'Individual tabular features vs. multimodal category aggregates',
        fontsize=12, fontweight='bold', pad=14,
    )

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.set_tick_params(labelsize=9)

    # extend x-axis for value labels
    ax.set_xlim(0, x_max * 1.18)
    ax.set_ylim(-0.6, n - 0.4)

    # legend
    legend_handles = [
        mpatches.Patch(color=COLOR_BASE,      label='Individual tabular feature'),
        mpatches.Patch(color=COLOR_SENTIMENT, label=LABEL_SENTIMENT),
        mpatches.Patch(color=COLOR_VISUAL,    label=LABEL_VISUAL),
        mpatches.Patch(color=COLOR_TFIDF,     label=LABEL_TFIDF),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8.5,
              framealpha=0.85, edgecolor='#cccccc')

    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='#e0e0e0', alpha=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {output_path}")


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
    X, feature_names, tfidf_tokens = build_features(train_df)
    y = train_df['AdoptionSpeed'].values

    total_features = len(feature_names) + N_TFIDF
    assert X.shape[1] == total_features, \
        f"Expected {total_features} columns, got {X.shape[1]}"
    print(f"Feature matrix shape: {X.shape}  ({len(feature_names)} tabular + {N_TFIDF} TF-IDF)")

    print("\nFitting Exp_36 GBR (n_estimators=200, lr=0.05, depth=4, subsample=0.8, leaf=10) …")
    model = fit_model(X, y)
    print("Fit complete.")

    importances = model.feature_importances_
    importance_dict = aggregate_importances(importances, feature_names, tfidf_tokens)

    print("\nAggregated importance summary:")
    for label, val in sorted(importance_dict.items(), key=lambda x: -x[1]):
        tag = ' ★ aggregate' if label in (LABEL_SENTIMENT, LABEL_VISUAL, LABEL_TFIDF) else ''
        print(f"  {label:<38s}  {val:.5f}{tag}")

    plot_importance(importance_dict, output_path='feature_importance.png')


if __name__ == '__main__':
    main()
