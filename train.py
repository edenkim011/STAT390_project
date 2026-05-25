# GBR + full feature set + within-fold TF-IDF — 5-fold CV evaluation (best: Exp_36)

import json
import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from prepare import evaluate_model


def load_metadata(pet_ids, metadata_dir='data/train_metadata', agg='mean'):
    """Extract color, crop, and label features from per-photo metadata JSONs.
    agg: 'mean' | 'max' | 'first'
    """
    from collections import defaultdict
    pet_photos = defaultdict(list)
    for fname in os.listdir(metadata_dir):
        if not fname.endswith('.json'):
            continue
        pid = fname.rsplit('-', 1)[0]
        pet_photos[pid].append(fname)

    records = []
    for pid in pet_ids:
        photos = sorted(pet_photos.get(pid, []))
        if not photos:
            records.append({'PetID': pid})
            continue

        if agg == 'first':
            photos = photos[:1]

        per_photo = []
        for fname in photos:
            with open(os.path.join(metadata_dir, fname)) as f:
                d = json.load(f)
            colors = d.get('imagePropertiesAnnotation', {}) \
                      .get('dominantColors', {}).get('colors', [])
            crops  = d.get('cropHintsAnnotation', {}).get('cropHints', [])
            labels = d.get('labelAnnotations', [])

            top3_pf = sum(c['pixelFraction'] for c in colors[:3]) if colors else 0
            label_scores = [l['score'] for l in labels]

            per_photo.append({
                'top_color_score':    colors[0]['score']         if colors else 0,
                'top_color_pixel_frac': colors[0]['pixelFraction'] if colors else 0,
                'color_count':        len(colors),
                'top3_pixel_frac_sum': top3_pf,
                'crop_confidence':    crops[0].get('confidence', 0)         if crops else 0,
                'crop_importance':    crops[0].get('importanceFraction', 0)  if crops else 0,
                'top_label_score':    labels[0]['score']           if labels else 0,
                'label_count':        len(labels),
                'label_score_mean':   np.mean(label_scores)       if label_scores else 0,
            })

        df_p = pd.DataFrame(per_photo)
        if agg == 'max':
            row = df_p.max().to_dict()
        else:  # mean or first
            row = df_p.mean().to_dict()
        row['PetID'] = pid
        records.append(row)

    return pd.DataFrame(records).fillna(0)


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
        entities = d.get('entities', [])
        doc = d.get('documentSentiment', {})
        scores = [s['sentiment']['score'] for s in sentences]
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


def run_baseline():
    start_time = time.time()

    train_df = pd.read_csv('data/working_train.csv')

    # load and merge sentiment features
    sent_df = load_sentiment(train_df['PetID'].tolist())
    train_df = train_df.merge(sent_df, on='PetID', how='left').fillna(0)

    # visual metadata: mean-aggregated across all photos per pet
    meta_df = load_metadata(train_df['PetID'].tolist(), agg='mean')
    train_df = train_df.merge(meta_df, on='PetID', how='left').fillna(0)

    base_features = ['Age', 'Fee', 'Type', 'Gender', 'Health', 'Vaccinated',
                     'Dewormed', 'Sterilized', 'PhotoAmt', 'Quantity',
                     'MaturitySize', 'FurLength', 'Breed1', 'Color1',
                     'Breed2', 'Color2', 'Color3', 'State', 'VideoAmt']
    sentiment_features = ['doc_score', 'doc_magnitude',
                          'sent_count', 'sent_score_mean', 'sent_mag_mean',
                          'sent_score_std', 'sent_score_max', 'sent_score_min', 'sent_mag_max',
                          'entity_count', 'top_salience']
    # visual features: full set (color + crop + label), mean-aggregated (Exp_20 best)
    visual_features = ['top_color_score', 'top_color_pixel_frac',
                       'color_count', 'top3_pixel_frac_sum',
                       'crop_confidence', 'crop_importance',
                       'top_label_score', 'label_count', 'label_score_mean']
    features = base_features + sentiment_features + visual_features
    target = 'AdoptionSpeed'

    X = train_df[features].values
    y = train_df[target].values
    descriptions = train_df['Description'].apply(lambda x: x if isinstance(x, str) else '').values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx].copy(), X[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train[:, :2] = scaler.fit_transform(X_train[:, :2])
        X_val[:, :2]   = scaler.transform(X_val[:, :2])

        # within-fold TF-IDF on raw description text
        tfidf = TfidfVectorizer(max_features=15, stop_words='english', sublinear_tf=True)
        tfidf_train = tfidf.fit_transform(descriptions[train_idx]).toarray()
        tfidf_val   = tfidf.transform(descriptions[val_idx]).toarray()
        X_train = np.column_stack([X_train, tfidf_train])
        X_val   = np.column_stack([X_val,   tfidf_val])

        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, min_samples_leaf=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        fold_rmse = evaluate_model(y_val, preds)
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold} RMSE: {fold_rmse:.4f}")

    cv_rmse = np.mean(fold_rmses)
    cv_std  = np.std(fold_rmses)

    end_time = time.time()
    duration = round(end_time - start_time, 4)

    print(f"5-fold CV RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")
    print(f'Runtime: {duration} seconds')

    return cv_rmse


if __name__ == '__main__':
    run_baseline()


