"""Experiment 2: models on the feature subsets described by filt_feature.csv.

filt_feature.csv marks every feature with 1 or 2 and gives the window
parameter that belongs to that feature.

  * features marked 1 -> their parameter is fixed (taken from 3exp_opt_1.csv
    when that file provides it, otherwise the mean of the range [7, Y]);
  * features marked 2 -> their parameter is searched with the same adaptive
    pattern search as in experiment 1.

Three models are compared: the features marked 1, the features marked 2 and
all the features of the table together.
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
from dataset import build_samples
from featurize import FeatureBuilder, prepare_samples

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42

FEATURE_PARAM = {
    "user_tx_cnt": "n", "user_avg_active_day_gap": "n",
    "item_tx_cnt": "m", "item_avg_active_day_gap": "m",
    "item_repeat_purchase_rate": "m",
    "cat_tx_cnt": "k", "viscat_tx_cnt": "x", "vi_tx_cnt": "q",
    "vc_tx_cnt": "s", "ic_tx_cnt": "z", "ic_view_to_tx_conv": "z",
    "ii_co_purchase": "u", "ii_co_purchase_cnt": "u",
}


def parse_filt(path):
    """-> DataFrame[feature, marker, param_label]"""
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}
    feat_col = next((cols[c] for c in cols
                     if c in ("feature", "feature_name", "name", "признак")), None)
    if feat_col is None:
        feat_col = raw.columns[0]
    marker_col = None
    for c in raw.columns:
        vals = raw[c].astype(str).str.strip()
        if set(vals.unique()) <= {"1", "2"} and len(vals.unique()) == 2:
            marker_col = c
            break
    if marker_col is None:
        for key in ("group", "marker", "mark", "type", "use", "модель"):
            if key in cols:
                marker_col = cols[key]
                break
    param_col = None
    for key in ("param", "parameter", "window", "window_param", "param_name",
                "окно", "параметр"):
        if key in cols:
            param_col = cols[key]
            break
    out = pd.DataFrame({
        "feature": raw[feat_col].astype(str).str.strip(),
        "marker": raw[marker_col].astype(str).str.strip().astype(int) if marker_col else 1,
    })
    out["param_label"] = (raw[param_col].astype(str).str.strip()
                          if param_col else out["feature"].map(
                              lambda f: FEATURE_PARAM.get(f, "")))
    out["builder_param"] = out["feature"].map(
        lambda f: FEATURE_PARAM.get(f, out["param_label"].get(f, "")))
    print(out.to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filt", default="filt_feature.csv")
    ap.add_argument("--opt1", default="3exp_opt_1.csv")
    ap.add_argument("--opt-params", default="1exp_opt_1.csv",
                    help="optimal n, m, s, z of the first experiment")
    ap.add_argument("--out-opt", default="2exp_opt.csv")
    ap.add_argument("--out-metrics", default="2exp_models_comparison.csv")
    ap.add_argument("--g6-stride", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config()
    lo, hi = cfg["param_min"], cfg["param_max"]
    mean = cfg["param_mean"]

    filt = parse_filt(args.filt)
    filt = filt[filt["builder_param"] != ""].reset_index(drop=True)

    # ---- parameters of the features marked 1 ------------------------------
    fixed = {}
    opt1 = None
    if os.path.exists(args.opt1):
        opt1 = pd.read_csv(args.opt1)
        print("\n3exp_opt_1.csv:\n", opt1.to_string(index=False))
    opt_params = {}
    if os.path.exists(args.opt_params):
        opt_params = pd.read_csv(args.opt_params).iloc[0].to_dict()
        print("\n1exp_opt_1.csv:\n", opt_params)

    for _, r in filt.iterrows():
        if r["marker"] == 1:
            p = r["builder_param"]
            val = None
            if opt1 is not None and p in opt1.columns:
                val = int(opt1[p].iloc[0])
            elif p in opt_params:
                val = int(opt_params[p])
            fixed[p] = val if val is not None else mean
    print("\nfixed parameters (marker 1):", fixed)

    # ---- parameters to search (marker 2) ----------------------------------
    search_params = sorted({r["builder_param"] for _, r in filt.iterrows()
                             if r["marker"] == 2})
    print("searched parameters (marker 2):", search_params)

    samples = build_samples(prepare_samples(), cfg)
    y = samples["target"].to_numpy()
    days = np.sort(samples["day"].unique())
    cut_tr, cut_va = days[int(len(days) * 0.70)], days[int(len(days) * 0.85)]
    tr = (samples["day"] < cut_tr).to_numpy()
    va = ((samples["day"] >= cut_tr) & (samples["day"] < cut_va)).to_numpy()

    fb = FeatureBuilder(samples, cfg, g6_stride=args.g6_stride)
    cache = {}

    def build(params, needed):
        cols = {}
        for p in sorted(needed):
            key = (p, params[p])
            if key not in cache:
                cache[key] = getattr(fb, fb.GROUP_FUNCS[p])(params[p])
            cols.update(cache[key])
        return cols

    needed_all = sorted(set(filt["builder_param"]))
    feats_all = list(filt["feature"])
    feats_1 = list(filt.loc[filt["marker"] == 1, "feature"])
    feats_2 = list(filt.loc[filt["marker"] == 2, "feature"])

    def objective(point):
        params = dict(fixed)
        params.update(dict(zip(search_params, point)))
        cols = build(params, needed_all)
        X = np.column_stack([cols[f] for f in feats_all])
        model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.08,
                                   num_leaves=63, min_child_samples=100,
                                   subsample=0.9, subsample_freq=1,
                                   colsample_bytree=0.9, reg_lambda=1.0,
                                   random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:, 1]
        return roc_auc_score(y[va], p)

    best_point, best_auc = {}, None
    if search_params:
        import importlib
        ps = importlib.import_module("05_exp1").pattern_search
        start = {p: mean for p in search_params}
        best_point, best_auc, history = ps(objective, search_params, start, lo, hi)
        with open("2exp_search_history.json", "w", encoding="utf-8") as f:
            json.dump({"best_auc": best_auc, "point": best_point,
                       "history": history}, f, indent=1)
    params = dict(fixed)
    params.update(best_point)
    print("\nfinal parameters:", params)

    pd.DataFrame([params]).to_csv(args.out_opt, index=False, lineterminator="\r\n")

    # ---- compare the models ------------------------------------------------
    rows = []
    for name, feats in (("model_features_marked_1", feats_1),
                        ("model_features_marked_2", feats_2),
                        ("model_features_marked_1_and_2", feats_all)):
        if not feats:
            continue
        cols = build(params, sorted({FEATURE_PARAM[f] for f in feats}))
        X = np.column_stack([cols[f] for f in feats])
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08,
                                   num_leaves=63, min_child_samples=100,
                                   subsample=0.9, subsample_freq=1,
                                   colsample_bytree=0.9, reg_lambda=1.0,
                                   random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:, 1]
        pred = (p >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y[va], pred,
                                                            average="binary",
                                                            zero_division=0)
        rows.append({
            "model": name,
            "features": ", ".join(feats),
            "n_features": len(feats),
            "params": ", ".join(f"{p}={params[p]}" for p in
                                sorted({FEATURE_PARAM[f] for f in feats})),
            "roc_auc": roc_auc_score(y[va], p),
            "pr_auc": average_precision_score(y[va], p),
            "log_loss": log_loss(y[va], p),
            "accuracy": accuracy_score(y[va], pred),
            "precision": prec, "recall": rec, "f1": f1,
        })
    md = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    md.to_csv(args.out_metrics, index=False, lineterminator="\r\n")
    print("\n", md.to_string(index=False))
    print("\nwrote", args.out_opt, args.out_metrics)


if __name__ == "__main__":
    main()
