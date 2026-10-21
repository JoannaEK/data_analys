"""Build the file model_prosto.csv: the features of every model, the window
parameter of every feature, the source of the value and the metrics of the
model.

The file is long (one row per model x evaluation x feature), so it can be
filtered or pivoted in any way.

Columns
-------
model               the name of the model
evaluation          the protocol the metrics were measured with
n_features          the number of features of the model
n_rows              the number of rows of the evaluation
pos_rate            the share of the positive rows
roc_auc pr_auc log_loss accuracy precision recall f1
                    the metrics of the evaluation
threshold           the decision threshold (found on the OOF of the CV and
                    transferred to the holdout)
f1_max threshold_max
                    the best F1 of the evaluation and its threshold
feature             the name of the feature
feature_description the full name of the feature with its arguments
feature_group       the group of the specification the feature belongs to
param               the window parameter of the feature (n, m, k, x, q, s, z, u)
window_days         the length of the window that was used, in days
source              where the value of the feature comes from
gain                the gain of the feature in the model (LightGBM)
gain_share          the share of the gain of the model, %
"""
import os

import pandas as pd

from config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))

# feature -> window parameter
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

# the group of the specification every feature belongs to
FEATURE_GROUP = {
    "user_tx_cnt": "1 (visitor)",
    "user_avg_active_day_gap": "1 (visitor)",
    "item_tx_cnt": "2 (itemid)",
    "item_avg_active_day_gap": "2 (itemid)",
    "item_repeat_purchase_rate": "2 (itemid)",
    "cat_tx_cnt": "3.1 (item_cat)",
    "viscat_tx_cnt": "3.2 (visitor_cat)",
    "vi_tx_cnt": "4 (visitor, itemid)",
    "vc_tx_cnt": "5.1 (visitor, item_cat)",
    "ic_tx_cnt": "5.2 (itemid, visitor_cat)",
    "ic_view_to_tx_conv": "5.2 (itemid, visitor_cat)",
    "ii_co_purchase": "6 (itemid, itemid)",
    "ii_co_purchase_cnt": "6 (itemid, itemid)",
}

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
    "vi_tx_cnt": "vi_tx_cnt(v, i, t, q) - the number of transactions of the pair "
                 "(visitor v, item i) in the window W(t, q)",
    "vc_tx_cnt": "vc_tx_cnt(v, c, t, s) - the number of transactions of the pair "
                 "(visitor v, item category c) in the window W(t, s)",
    "ic_tx_cnt": "ic_tx_cnt(i, c, t, z) - the number of transactions of the pair "
                 "(item i, visitor category c) in the window W(t, z)",
    "ic_view_to_tx_conv": "ic_view_to_tx_conv(i, c, t, z) - the conversion of the "
                          "views into transactions for the pair (item i, visitor "
                          "category c) in the window W(t, z)",
    "ii_co_purchase": "ii_co_purchase(i, j, t, u) - 1 if the pair of items "
                      "(i, j) was bought by at least two different visitors in "
                      "the window W(t, u), 0 otherwise",
    "ii_co_purchase_cnt": "ii_co_purchase_cnt(i, t, u) - the number of the items "
                          "j that were co-purchased with the item i in the "
                          "window W(t, u)",
}

# the window the published table of the repository was built with: the start
# window of its own experiment 1 (the mean of the range [7, 738])
REPO_TABLE_WINDOW = 372.606
# the optimal windows of the groups 1, 2, 5.1, 5.2 (1exp_opt_1.csv), rounded
REPO_OPTIMAL = {"n": 198, "m": 518, "s": 13, "z": 348}
REPO_OPTIMAL_EXACT = {"n": 197.606, "m": 517.606, "s": 12.606, "z": 347.606}

PART2_FEATURES = {"cat_tx_cnt", "viscat_tx_cnt", "vi_tx_cnt", "ii_co_purchase",
                  "ii_co_purchase_cnt"}


def read_wide(path):
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    out = {}
    for c in d.columns:
        try:
            out[str(c).strip()] = float(d[c].iloc[0])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def window_of(feature, part2_windows):
    """-> (param, window in days, source of the value)"""
    base = feature[:-4] if feature.endswith("_opt") else feature
    p = FEATURE_PARAM[base]
    if feature.endswith("_opt"):
        return p, REPO_OPTIMAL[p], ("features_part_1_local_opt.csv - recomputed "
                                    f"locally with the optimal window of the "
                                    f"group ({REPO_OPTIMAL_EXACT[p]} days)")
    if base in PART2_FEATURES:
        return p, part2_windows.get(p), ("features_part_2.csv - the optimal "
                                         "window of experiment 1")
    if base == "ic_view_to_tx_conv":
        return p, REPO_TABLE_WINDOW, ("feature_1.csv of the repository - not "
                                      "reproducible locally (there is no view "
                                      "log)")
    return p, REPO_TABLE_WINDOW, ("feature_1.csv of the repository - the start "
                                   f"window of its experiment 1 (the optimal "
                                   f"window of the group is "
                                   f"{REPO_OPTIMAL_EXACT[p]} days)")


def main(out="model_prosto.csv"):
    part2_windows = read_wide(os.path.join(HERE, "1exp_opt_2.csv"))
    metrics = pd.read_csv(os.path.join(HERE, "models_comparison.csv"))
    feats = pd.read_csv(os.path.join(HERE, "models_features.csv"))
    imp = pd.read_csv(os.path.join(HERE, "models_feature_importance.csv"))

    rows = []
    for _, m in metrics.iterrows():
        name = m["model"]
        use = feats.loc[feats["model"] == name, "features"]
        flist = [x.strip() for x in str(use.iloc[0]).split(",")] if len(use) else []
        g = imp[imp["model"] == name].set_index("feature")["gain"]
        total = float(g.sum()) if len(g) else 0.0
        for f in flist:
            p, w, src = window_of(f, part2_windows)
            gain = float(g.get(f, 0.0))
            rows.append({
                "model": name,
                "evaluation": m["evaluation"],
                "n_features": int(m["n_features"]),
                "n_rows": int(m["n_rows"]),
                "pos_rate": round(float(m["pos_rate"]), 6),
                "roc_auc": round(float(m["roc_auc"]), 6),
                "pr_auc": round(float(m["pr_auc"]), 6),
                "log_loss": round(float(m["log_loss"]), 6),
                "accuracy": round(float(m["accuracy"]), 6),
                "precision": round(float(m["precision"]), 6),
                "recall": round(float(m["recall"]), 6),
                "f1": round(float(m["f1"]), 6),
                "threshold": round(float(m["threshold"]), 6),
                "f1_max": round(float(m["f1_max"]), 6),
                "threshold_max": round(float(m["threshold_max"]), 6),
                "feature": f,
                "feature_description": FEATURE_DESC.get(
                    f[:-4] if f.endswith("_opt") else f, f),
                "feature_group": FEATURE_GROUP.get(
                    f[:-4] if f.endswith("_opt") else f, ""),
                "param": p,
                "window_days": w,
                "source": src,
                "gain": round(gain, 1),
                "gain_share": round(100 * gain / total, 2) if total else 0.0,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "evaluation", "gain_share"],
                         ascending=[True, True, False])
    df.to_csv(os.path.join(HERE, out), index=False, lineterminator="\r\n")
    print(f"wrote {out} ({len(df)} rows, {df['model'].nunique()} models)")
    print(df.groupby("model")["feature"].nunique().to_string())


if __name__ == "__main__":
    main()
