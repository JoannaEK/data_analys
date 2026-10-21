"""Experiment 2: models on the feature subsets described by filt_feature.csv.

filt_feature.csv marks every feature with 1 or 2 and gives the window
parameter that belongs to that feature (the column new_param).

  * a feature marked 1 -> its parameter is fixed: the value of
    3exp_opt_1.csv for that parameter (n1, s1, ...) or, when the file does
    not provide it, the mean of the range [7, Y];
  * a feature marked 2 -> its parameter is searched with the same adaptive
    pattern search as in experiment 1.

In experiment 2 every feature has its own window (the parameters of the
second experiment - n1, n2, x1, q1, s1 - are per feature, not per group),
so a group is computed once per window that is needed by its features.

Three models are compared: the features marked 1, the features marked 2 and
all the features of the table together.  The metrics are computed on the
validation window (70% .. 85% of the time axis); the rows with an identified
buyer are reported separately, because the target is only meaningful there.
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
    """-> DataFrame[feature, marker, param_label] with marker in {1, 2}."""
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}
    feat_col = cols.get("feature") or raw.columns[0]
    marker_col = None
    for key in ("needed", "marker", "mark", "group", "type", "use"):
        if key in cols:
            marker_col = cols[key]
            break
    param_col = None
    for key in ("new_param", "param", "parameter", "window", "window_param"):
        if key in cols:
            param_col = cols[key]
            break
    out = pd.DataFrame({
        "feature": raw[feat_col].astype(str).str.strip(),
        "marker": raw[marker_col].astype(str).str.strip().astype(int),
    })
    out["param_label"] = (raw[param_col].astype(str).str.strip()
                          if param_col is not None else "")
    out["builder_param"] = out["feature"].map(
        lambda f: FEATURE_PARAM.get(f, ""))
    out = out[out["marker"].isin([1, 2]) & (out["builder_param"] != "")]
    out = out.reset_index(drop=True)
    print(out.to_string(index=False))
    return out


def read_opt(path, key_col):
    """-> dict label -> window (days)"""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    key_col = key_col if key_col in d.columns else d.columns[0]
    val_col = ("optimal_window_days" if "optimal_window_days" in d.columns
               else d.columns[-1])
    out = {}
    for _, r in d.iterrows():
        try:
            out[str(r[key_col]).strip()] = float(r[val_col])
        except (TypeError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filt", default="filt_feature.csv")
    ap.add_argument("--opt1", default="3exp_opt_1.csv")
    ap.add_argument("--opt-params", default="1exp_opt_1.csv")
    ap.add_argument("--out-opt", default="2exp_opt.csv")
    ap.add_argument("--out-metrics", default="2exp_models_comparison.csv")
    ap.add_argument("--out-history", default="2exp_search_history.json")
    ap.add_argument("--g6-stride", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config()
    lo, hi = cfg["param_min"], cfg["param_max"]
    mean = cfg["param_mean"]

    filt = parse_filt(args.filt)
    opt1 = read_opt(args.opt1, "param")
    opt_params = read_opt(args.opt_params, "param")
    print("\n3exp_opt_1.csv :", opt1)
    print("1exp_opt_1.csv :", opt_params)

    # ---- fixed parameters (marker 1) ---------------------------------------
    fixed = {}
    for _, r in filt[filt["marker"] == 1].iterrows():
        label = r["param_label"] or r["builder_param"]
        val = opt1.get(label)
        if val is None:
            val = opt_params.get(label)
        fixed[label] = int(round(val)) if val is not None else mean
    print("\nfixed parameters (marker 1):", fixed)

    # ---- parameters to search (marker 2) -----------------------------------
    search_labels = sorted({r["param_label"] or r["builder_param"]
                            for _, r in filt[filt["marker"] == 2].iterrows()})
    print("searched parameters (marker 2):", search_labels)

    samples = build_samples(prepare_samples(), cfg)
    y = samples["target"].to_numpy()
    known = (samples["visitor"] != "").to_numpy()
    days = np.sort(samples["day"].unique())
    cut_tr, cut_va = days[int(len(days) * 0.70)], days[int(len(days) * 0.85)]
    tr = (samples["day"] < cut_tr).to_numpy()
    va = ((samples["day"] >= cut_tr) & (samples["day"] < cut_va)).to_numpy()
    print(f"train rows {tr.sum()}  valid rows {va.sum()}")

    fb = FeatureBuilder(samples, cfg, g6_stride=args.g6_stride)
    cache = {}

    def group_cols(group, window):
        """The features of one group computed with one window."""
        key = (group, window)
        if key not in cache:
            cache[key] = getattr(fb, fb.GROUP_FUNCS[group])(window)
        return cache[key]

    def feature_window(label, value):
        """window of one feature: the label of the second experiment decides
        which window of which group is used."""
        for _, r in filt.iterrows():
            if (r["param_label"] or r["builder_param"]) == label:
                return r["builder_param"], value
        raise KeyError(label)

    def matrix(params):
        """-> (X, feature names) for the given parameter values per label."""
        cols, names = [], []
        for _, r in filt.iterrows():
            label = r["param_label"] or r["builder_param"]
            group = r["builder_param"]
            w = params[label]
            gc = group_cols(group, w)
            cols.append(gc[r["feature"]])
            names.append(r["feature"])
        return np.column_stack(cols), names

    params = dict(fixed)
    for label in search_labels:
        params[label] = mean

    def objective(point):
        p = dict(params)
        p.update(dict(zip(search_labels, point)))
        X, _ = matrix(p)
        model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.08,
                                   num_leaves=63, min_child_samples=100,
                                   subsample=0.9, subsample_freq=1,
                                   colsample_bytree=0.9, reg_lambda=1.0,
                                   random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(X[tr], y[tr])
        p_va = model.predict_proba(X[va])[:, 1]
        yv = y[va]
        mk = known[va]
        objective.auc_known = (roc_auc_score(yv[mk], p_va[mk])
                                if len(np.unique(yv[mk])) > 1 else float("nan"))
        return roc_auc_score(yv, p_va)

    best_point, best_auc = {}, float("nan")
    if search_labels:
        import importlib
        ps = importlib.import_module("05_exp1").pattern_search
        start = {label: mean for label in search_labels}
        best_point, best_auc, history = ps(objective, search_labels, start,
                                            lo, hi)
        with open(args.out_history, "w", encoding="utf-8") as f:
            json.dump({"best_auc": best_auc, "point": best_point,
                       "history": history}, f, indent=1)
    params.update(best_point)
    print("\nfinal parameters:", params)

    rows = []
    for name, feats in (("model_features_marked_1",
                         list(filt.loc[filt["marker"] == 1, "feature"])),
                        ("model_features_marked_2",
                         list(filt.loc[filt["marker"] == 2, "feature"])),
                        ("model_features_marked_1_and_2",
                         list(filt["feature"]))):
        if not feats:
            continue
        sub = filt[filt["feature"].isin(feats)]
        cols, names = [], []
        for _, r in sub.iterrows():
            label = r["param_label"] or r["builder_param"]
            gc = group_cols(r["builder_param"], params[label])
            cols.append(gc[r["feature"]])
            names.append(r["feature"])
        X = np.column_stack(cols)
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08,
                                   num_leaves=63, min_child_samples=100,
                                   subsample=0.9, subsample_freq=1,
                                   colsample_bytree=0.9, reg_lambda=1.0,
                                   random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(X[tr], y[tr])
        p_va = model.predict_proba(X[va])[:, 1]
        yv = y[va]
        mk = known[va]
        pred = (p_va >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(yv, pred,
                                                            average="binary",
                                                            zero_division=0)
        pk = (p_va[mk] >= 0.5).astype(int)
        prec_k, rec_k, f1_k, _ = precision_recall_fscore_support(
            yv[mk], pk, average="binary", zero_division=0)
        rows.append({
            "model": name,
            "features": ", ".join(names),
            "n_features": len(names),
            "params": ", ".join(
                f"{r['param_label'] or r['builder_param']}="
                f"{params[r['param_label'] or r['builder_param']]}"
                for _, r in sub.iterrows()),
            "roc_auc": roc_auc_score(yv, p_va),
            "pr_auc": average_precision_score(yv, p_va),
            "log_loss": log_loss(yv, p_va),
            "accuracy": accuracy_score(yv, pred),
            "precision": prec, "recall": rec, "f1": f1,
            "roc_auc_known_buyer": roc_auc_score(yv[mk], p_va[mk]),
            "pr_auc_known_buyer": average_precision_score(yv[mk], p_va[mk]),
            "f1_known_buyer": f1_k,
            "n_rows_valid": int(va.sum()),
            "n_rows_valid_known_buyer": int(mk.sum()),
        })
    md = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    md.to_csv(args.out_metrics, index=False, lineterminator="\r\n")
    pd.DataFrame([params]).to_csv(args.out_opt, index=False,
                                   lineterminator="\r\n")
    print("\n", md.to_string(index=False))
    print("\nwrote", args.out_opt, args.out_metrics)


if __name__ == "__main__":
    main()
