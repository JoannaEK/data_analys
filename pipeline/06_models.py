"""Final stage: train the models and compare them by metrics.

Feature sources
---------------
groups 1, 2, 5.1, 5.2  -> features_part_1.csv (repository) or the local
                         fallback 04_features_part1.py
groups 3.1, 3.2, 4      -> features_part_2.csv (this pipeline)
group 6                -> features_part_2_group6.csv + the row-grain
                         projection ii_co_purchase_cnt inside features_part_2.csv

Every model is described by the list of features it uses and by the window
parameter of each of those features; that description is written next to the
metrics.
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             log_loss, precision_recall_fscore_support,
                             roc_auc_score)

from config import load_config
from dataset import build_samples, load_retail

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# feature -> window parameter (see the specification of the feature groups)
FEATURE_PARAM = {
    "user_tx_cnt": "n",
    "user_avg_active_day_gap": "n",
    "item_tx_cnt": "m",
    "item_avg_active_day_gap": "m",
    "item_repeat_purchase_rate": "m",
    "cat_tx_cnt": "k",
    "viscat_tx_cnt": "x",
    "vi_tx_cnt": "q",
    "vc_tx_cnt": "s",
    "ic_tx_cnt": "z",
    "ic_view_to_tx_conv": "z",
    "ii_co_purchase": "u",
    "ii_co_purchase_cnt": "u",
}

GROUP_FEATURES = {
    "group_1_user": ["user_tx_cnt", "user_avg_active_day_gap"],
    "group_2_item": ["item_tx_cnt", "item_avg_active_day_gap",
                     "item_repeat_purchase_rate"],
    "group_3.1_item_cat": ["cat_tx_cnt"],
    "group_3.2_visitor_cat": ["viscat_tx_cnt"],
    "group_4_visitor_item": ["vi_tx_cnt"],
    "group_5.1_visitor_item_cat": ["vc_tx_cnt"],
    "group_5.2_item_visitor_cat": ["ic_tx_cnt", "ic_view_to_tx_conv"],
    "group_6_item_item": ["ii_co_purchase", "ii_co_purchase_cnt"],
}

# full names of the features as they are defined in the specification
FEATURE_DESC = {
    "user_tx_cnt": "user_tx_cnt(v, t, n) - the number of transactions of the "
                   "visitor v in the window W(t, n)",
    "user_avg_active_day_gap": "user_avg_active_day_gap(v, t, n) - the average "
                               "gap between the active days of the visitor v in "
                               "the window W(t, n)",
    "item_tx_cnt": "item_tx_cnt(i, t, m) - the number of transactions of the "
                   "item i in the window W(t, m)",
    "item_avg_active_day_gap": "item_avg_active_day_gap(i, t, m) - the average "
                               "gap between the active days of the item i in the "
                               "window W(t, m)",
    "item_repeat_purchase_rate": "item_repeat_purchase_rate(i, t, m) - the share "
                                 "of the users who bought the item i on at least "
                                 "two distinct days of the window W(t, m)",
    "cat_tx_cnt": "cat_tx_cnt(c, t, k) - the number of transactions of the item "
                  "category c in the window W(t, k)",
    "viscat_tx_cnt": "viscat_tx_cnt(c, t, x) - the number of transactions of the "
                      "visitor category c in the window W(t, x)",
    "vi_tx_cnt": "vi_tx_cnt(v, i, t, q) - the number of transactions of the "
                 "visitor v with the item i in the window W(t, q)",
    "vc_tx_cnt": "vc_tx_cnt(v, c, t, s) - the number of transactions of the "
                 "visitor v with the items of the category c in the window "
                 "W(t, s)",
    "ic_tx_cnt": "ic_tx_cnt(i, c, t, z) - the number of transactions of the "
                 "item i by the visitors of the category c in the window "
                 "W(t, z)",
    "ic_view_to_tx_conv": "ic_view_to_tx_conv(i, c, t, z) - the conversion of "
                          "the views of the item i into purchases by the visitors "
                          "of the category c in the window W(t, z)",
    "ii_co_purchase": "ii_co_purchase(i, j, t, u) - 1 if the pair of items "
                      "(i, j) was bought by at least two different visitors in "
                      "the window W(t, u), 0 otherwise",
    "ii_co_purchase_cnt": "ii_co_purchase_cnt(i, t, u) - the number of items j "
                          "with ii_co_purchase(i, j, t, u) = 1 in the window "
                          "W(t, u)",
}


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    f1_max, thr_max = f1_at_best(y_true, y_prob)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": threshold,
        "f1_max": f1_max,
        "threshold_max": thr_max,
        "pos_rate_pred": float(y_pred.mean()),
    }


def best_f1_threshold(y_true, y_prob):
    """Threshold that maximises F1 (used for a fair comparison of the models)."""
    order = np.argsort(-y_prob)
    ys = y_true[order]
    ps = y_prob[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    n_pos = ys.sum()
    if n_pos == 0:
        return 0.5
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    # only thresholds between two consecutive predictions are valid
    ok = np.ones(len(f1), dtype=bool)
    ok[:-1] = ps[:-1] != ps[1:]
    f1 = np.where(ok, f1, -1)
    i = int(np.argmax(f1))
    return float(min(1.0, max(0.0, ps[i])))


def f1_at_best(y_true, y_prob):
    """The best F1 that is reachable on the evaluation set itself.

    It is an optimistic (oracle) value: it shows the ceiling of the model and
    how far the threshold transferred from the cross-validation is from it.
    """
    thr = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                  average="binary",
                                                  zero_division=0)
    return float(f1), thr


def train_model(X, y, tr, te, params=None):
    p = dict(n_estimators=300, learning_rate=0.08, num_leaves=63,
             min_child_samples=100, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED,
             n_jobs=-1, verbose=-1)
    if params:
        p.update(params)
    model = lgb.LGBMClassifier(**p)
    model.fit(X[tr], y[tr])
    return model


def cv_metrics(X, y, day, n_folds=5, params=None):
    """Out-of-fold metrics of a model fitted on the whole data set.

    The folds are contiguous blocks of the time axis, so every row is used
    for training (in the other folds) and once for validation.
    """
    days = np.sort(np.unique(day))
    blocks = np.array_split(days, n_folds)
    oof = np.zeros(len(y), dtype=np.float64)
    for b in blocks:
        te = np.isin(day, b)
        tr = ~te
        model = train_model(X, y, tr, te, params)
        oof[te] = model.predict_proba(X[te])[:, 1]
    return evaluate(y, oof), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part2", default="features_part_2.csv")
    ap.add_argument("--part1", default="features_part_1.csv")
    ap.add_argument("--out-metrics", default="models_comparison.csv")
    ap.add_argument("--out-features", default="models_features.csv")
    ap.add_argument("--out-importances", default="models_feature_importance.csv")
    args = ap.parse_args()

    cfg = load_config()
    samples = build_samples(load_retail(), cfg)
    y = samples["target"].to_numpy()

    part2 = pd.read_csv(args.part2, parse_dates=["t"],
                         dtype={"visitor": "string", "itemid": "string"})
    part2["visitor"] = part2["visitor"].fillna("")
    part1_path = args.part1 if os.path.exists(args.part1) else None
    if part1_path:
        part1 = pd.read_csv(part1_path, parse_dates=["t"],
                            dtype={"visitor": "string", "itemid": "string"})
        part1["visitor"] = part1["visitor"].fillna("")
    else:
        part1 = None
    if part1 is None:
        print(f"!! {args.part1} not found - only groups 3.1/3.2/4/6 are used")

    key = ["t", "visitor", "itemid"]
    part2 = part2.drop_duplicates(subset=key, keep="first")
    if part1 is not None:
        part1 = part1.drop_duplicates(subset=key, keep="first")

    df = samples.merge(part2, on=key, how="left", suffixes=("", "_p2"),
                       validate="many_to_one")
    if part1 is not None:
        df = df.merge(part1, on=key, how="left", suffixes=("", "_p1"),
                      validate="many_to_one")
    if len(df) != len(samples):
        print(f"!! merged frame has {len(df)} rows for {len(samples)} samples")
    if "target" in df.columns and not (df["target"].to_numpy() == y).all():
        raise SystemExit("!! the merged frame is not aligned with the samples")
    print(f"merged frame: {len(df)} rows, alignment with the samples verified")

    days = np.sort(samples["day"].unique())
    cut = days[int(len(days) * 0.85)]
    tr = (samples["day"] < cut).to_numpy()
    te = (samples["day"] >= cut).to_numpy()

    available = [c for c in FEATURE_PARAM if c in df.columns]
    print("available features:", available)

    model_specs = []
    for gname, feats in GROUP_FEATURES.items():
        use = [f for f in feats if f in df.columns]
        if use:
            model_specs.append((f"model_{gname}", use))
    model_specs.append(("model_all_features", available))
    seen = set()
    model_specs = [(n, f) for n, f in model_specs
                   if not (tuple(f) in seen or seen.add(tuple(f)))]

    metrics, feat_desc, importances = [], [], []
    for name, feats in model_specs:
        X = df[feats].to_numpy(dtype=np.float64)

        # 1) out-of-fold predictions of the time-blocked CV: they define the
        #    decision threshold and the CV metrics
        cvm, oof = cv_metrics(X, y, samples["day"].to_numpy())
        thr = best_f1_threshold(y, oof)
        cvm = evaluate(y, oof, thr)
        cvm["model"] = name
        cvm["n_features"] = len(feats)
        cvm["evaluation"] = "5-fold time-blocked CV on the full data set"
        metrics.append(cvm)

        # 2) the final model: trained on the first 85% of the snapshots and
        #    evaluated on the last 15% with the threshold found on the CV
        model = train_model(X, y, tr, te)
        prob = model.predict_proba(X[te])[:, 1]
        m = evaluate(y[te], prob, thr)
        m["model"] = name
        m["n_features"] = len(feats)
        m["evaluation"] = "holdout (last 15% of the snapshots)"
        metrics.append(m)

        params_used = sorted({FEATURE_PARAM[f]: f for f in feats}.items())
        feat_desc.append({
            "model": name,
            "features": ", ".join(feats),
            "n_features": len(feats),
            "window_params": ", ".join(f"{p} (features: "
                                       f"{', '.join(f for f in feats if FEATURE_PARAM[f]==p)})"
                                       for p, _ in params_used),
            "feature_description": " | ".join(FEATURE_DESC[f] for f in feats),
            "feature_to_param": ", ".join(f"{f} -> {FEATURE_PARAM[f]}" for f in feats),
        })
        imp = model.booster_.feature_importance("gain")
        for f, v in zip(feats, imp):
            importances.append({"model": name, "feature": f, "gain": float(v)})
        print(f"{name:38s} holdout auc={m['roc_auc']:.5f}  cv auc={cvm['roc_auc']:.5f} "
              f"thr={thr:.4f} f1={m['f1']:.4f} logloss={m['log_loss']:.5f}", flush=True)

    md = pd.DataFrame(metrics)[["model", "evaluation", "n_features", "roc_auc",
                                "pr_auc", "log_loss", "accuracy", "precision",
                                "recall", "f1", "threshold", "f1_max",
                                "threshold_max"]]
    md = md.sort_values(["evaluation", "roc_auc"], ascending=[True, False])
    md.to_csv(args.out_metrics, index=False, lineterminator="\r\n")
    pd.DataFrame(feat_desc).to_csv(args.out_features, index=False, lineterminator="\r\n")
    pd.DataFrame(importances).to_csv(args.out_importances, index=False, lineterminator="\r\n")
    print("\n", md.to_string(index=False))
    print("\nwrote", args.out_metrics, args.out_features, args.out_importances)


if __name__ == "__main__":
    main()
