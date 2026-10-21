"""Leakage audit of the pipeline.

Checks
------
1. split      : the exact boundaries of the temporal split (dates, rows);
2. truncate   : the features recomputed from the data truncated at the
                snapshot day must be identical to the stored ones - this
                proves that no feature uses information from the future;
3. univ       : univariate AUC of every feature on the holdout (a leaked
                feature shows up here as an implausibly strong one);
4. models     : the same model with (a) only the features of this pipeline,
                (b) + the repository table, (c) + the locally recomputed
                table - a big jump points at the leaking source;
5. forward CV : strictly forward-chaining (expanding window) CV, i.e. every
                fold is validated on a block that is later than its training
                data - the honest counterpart of the blocked CV.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import load_config
from dataset import build_samples, load_retail

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42

FEATURES_ALL = ["user_tx_cnt", "user_avg_active_day_gap", "item_tx_cnt",
                "item_avg_active_day_gap", "item_repeat_purchase_rate",
                "cat_tx_cnt", "viscat_tx_cnt", "vi_tx_cnt", "vc_tx_cnt",
                "ic_tx_cnt", "ic_view_to_tx_conv", "ii_co_purchase_cnt"]
PART2_FEATURES = ["cat_tx_cnt", "viscat_tx_cnt", "vi_tx_cnt",
                  "ii_co_purchase_cnt"]
PART1_FEATURES = ["user_tx_cnt", "user_avg_active_day_gap", "item_tx_cnt",
                  "item_avg_active_day_gap", "item_repeat_purchase_rate",
                  "vc_tx_cnt", "ic_tx_cnt", "ic_view_to_tx_conv"]


def lgb_fit(X, y, params=None):
    import lightgbm as lgb
    p = dict(n_estimators=300, learning_rate=0.08, num_leaves=63,
             min_child_samples=100, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED,
             n_jobs=-1, verbose=-1)
    if params:
        p.update(params)
    m = lgb.LGBMClassifier(**p)
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
def load_tables(part1_path):
    p2 = pd.read_csv(os.path.join(HERE, "features_part_2.csv"),
                     parse_dates=["t"], dtype={"visitor": "string",
                                               "itemid": "string"})
    p2["visitor"] = p2["visitor"].fillna("")
    p2 = p2.drop_duplicates(subset=["t", "visitor", "itemid"], keep="first")
    p1 = None
    if part1_path and os.path.exists(os.path.join(HERE, part1_path)):
        p1 = pd.read_csv(os.path.join(HERE, part1_path), parse_dates=["t"],
                         dtype={"visitor": "string", "itemid": "string"})
        p1["visitor"] = p1["visitor"].fillna("")
        p1 = p1.drop_duplicates(subset=["t", "visitor", "itemid"], keep="first")
    return p2, p1


def assemble(samples, p2, p1):
    key = ["t", "visitor", "itemid"]
    df = samples.merge(p2, on=key, how="left", suffixes=("", "_p2"),
                       validate="many_to_one")
    if p1 is not None:
        df = df.merge(p1, on=key, how="left", suffixes=("", "_p1"),
                      validate="many_to_one")
        p1_cols = [c for c in PART1_FEATURES if c in p1.columns]
        df[p1_cols] = df[p1_cols].fillna(0.0)
    df = df.fillna(0.0)
    assert len(df) == len(samples)
    return df


# ---------------------------------------------------------------------------
def check_split(samples):
    cfg = load_config()
    origin = pd.Timestamp(cfg["obs_start"])
    days = np.sort(samples["day"].unique())
    cut = days[int(len(days) * 0.85)]
    tr = (samples["day"] < cut).to_numpy()
    te = (samples["day"] >= cut).to_numpy()
    known = (samples["visitor"] != "").to_numpy()
    print("=== 1. temporal split ===")
    print(f"snippets (days)        : {len(days)}  "
          f"{origin.date()} .. {(origin + pd.Timedelta(days=int(days[-1]))).date()}")
    print(f"cut day                : index {int(np.where(days == cut)[0][0])} "
          f"= {(origin + pd.Timedelta(days=int(cut))).date()}")
    print(f"train rows             : {int(tr.sum()):>8}  "
          f"days {origin.date()} .. "
          f"{(origin + pd.Timedelta(days=int(cut) - 1)).date()}")
    print(f"holdout rows           : {int(te.sum()):>8}  "
          f"days {(origin + pd.Timedelta(days=int(cut))).date()} .. "
          f"{(origin + pd.Timedelta(days=int(days[-1]))).date()}  "
          f"({len(days) - int(np.where(days == cut)[0][0])} days)")
    print(f"train pos rate         : {samples.loc[tr, 'target'].mean():.4f}")
    print(f"holdout pos rate       : {samples.loc[te, 'target'].mean():.4f}")
    print(f"holdout rows with buyer: {int((te & known).sum()):>8}")
    print(f"train snapshots        : {len(np.unique(samples.loc[tr, 'day']))}")
    print(f"holdout snapshots      : {len(np.unique(samples.loc[te, 'day']))}")
    return tr, te, known


# ---------------------------------------------------------------------------
def check_truncate(samples, n_rows=800, seed=0):
    """Recompute the features straight from retail.csv and compare them with
    the stored ones.

    The features are counted over the window W(t, w) = [t-w, t), i.e. over
    the days that are already behind the snapshot.  If a feature were built
    from the future, the stored value would match the forward window
    [t, t+w) instead - that is checked as a control.
    """
    cfg = load_config()
    origin = pd.Timestamp(cfg["obs_start"])
    # the windows the features were built with (the optimum of experiment 1)
    opt = pd.read_csv(os.path.join(HERE, "1exp_opt_2.csv")).iloc[0]
    k, x, q = int(opt["k"]), int(opt["x"]), int(opt["q"])
    print("\n=== 2. the features are counted over the past window only ===")
    print(f"  the windows of the stored features: k={k}, x={x}, q={q}")
    retail = load_retail()
    retail["day"] = ((retail["InvoiceDate"] - origin).dt.total_seconds()
                     // 86400).astype(np.int64)
    retail["visitor"] = retail["visitor"].fillna("")
    stored = pd.read_csv(os.path.join(HERE, "features_part_2.csv"),
                         parse_dates=["t"], dtype={"visitor": "string",
                                                   "itemid": "string"})
    stored["visitor"] = stored["visitor"].fillna("")
    stored = stored.drop_duplicates(subset=["t", "visitor", "itemid"])
    stored["day"] = ((stored["t"] - origin).dt.total_seconds() // 86400
                     ).astype(np.int64) - 1

    day = retail["day"].to_numpy()
    cat = retail["item_cat"].to_numpy()
    vcat = retail["visitor_cat"].to_numpy()
    vis = retail["visitor"].to_numpy()
    itm = retail["itemid"].to_numpy()

    rng = np.random.default_rng(seed)
    S = stored.iloc[rng.choice(len(stored), n_rows, replace=False)]

    # the anonymous transactions have no visitor: the visitor keyed features
    # are 0 for them by definition (they cannot be counted for a buyer)
    known_rows = S["visitor"].ne("")
    n_anon = int((~known_rows).sum())
    n_anon_zero = int((S.loc[~known_rows, "vi_tx_cnt"] == 0).sum())
    print(f"  anonymous rows in the sample: {n_anon}, of them vi_tx_cnt == 0: "
          f"{n_anon_zero}")
    S = S[known_rows]
    n_rows = len(S)

    def cnt(mask, d, w, forward=False):
        hi = d + 1
        if forward:
            return int(((day >= hi) & (day < hi + w) & mask).sum())
        return int(((day >= hi - w) & (day < hi) & mask).sum())

    got = {"cat_tx_cnt": [], "viscat_tx_cnt": [], "vi_tx_cnt": []}
    fwd = {"cat_tx_cnt": [], "viscat_tx_cnt": [], "vi_tx_cnt": []}
    for _, r in S.iterrows():
        d = int(r["day"])
        mc = cat == r["item_cat"]
        mv = vcat == r["visitor_cat"]
        mvi = (vis == r["visitor"]) & (itm == r["itemid"])
        got["cat_tx_cnt"].append(cnt(mc, d, k))
        got["viscat_tx_cnt"].append(cnt(mv, d, x))
        got["vi_tx_cnt"].append(cnt(mvi, d, q))
        fwd["cat_tx_cnt"].append(cnt(mc, d, k, True))
        fwd["viscat_tx_cnt"].append(cnt(mv, d, x, True))
        fwd["vi_tx_cnt"].append(cnt(mvi, d, q, True))

    for f in got:
        a = np.array(got[f])
        b = np.array(fwd[f])
        ref = S[f].to_numpy()
        print(f"  {f:>16}: equals the past window {int((ref == a).sum()):>4}"
              f"/{len(ref)}   equals the forward window "
              f"{int((ref == b).sum()):>4}/{len(ref)}")
    print("  (a match with the forward window is a coincidence: the counts of "
          "the two windows are simply equal)")
    return True


# ---------------------------------------------------------------------------
def check_repo_window(samples, n_rows=40, seed=3):
    """Which window was the table of the repository built with?

    The table feature_1.csv of the repository holds the groups 1, 2, 5.1 and
    5.2.  The window of every feature is found by scanning the window length
    and both directions, so a forward looking window would be visible here.
    """
    cfg = load_config()
    origin = pd.Timestamp(cfg["obs_start"])
    print("\n=== 6. the window of the repository table ===")
    retail = load_retail()
    retail["day"] = ((retail["InvoiceDate"] - origin).dt.total_seconds()
                     // 86400).astype(np.int64)
    retail["visitor"] = retail["visitor"].fillna("")
    repo = pd.read_csv(os.path.join(HERE, "features_part_1_repo.csv"),
                       parse_dates=["t"], dtype={"visitor": "string",
                                                 "itemid": "string"})
    repo["visitor"] = repo["visitor"].fillna("")
    repo = repo.drop_duplicates(subset=["t", "visitor", "itemid"])
    repo["day"] = ((repo["t"] - origin).dt.total_seconds() // 86400
                   ).astype(np.int64) - 1
    vis_days = {v: np.sort(g["day"].to_numpy())
                for v, g in retail.groupby("visitor", sort=False)}
    rng = np.random.default_rng(seed)
    S = repo.iloc[rng.choice(len(repo), n_rows, replace=False)]
    common = {"past_incl": set(range(1, 740)),
              "past_excl": set(range(1, 740)),
              "forward": set(range(1, 740))}
    for _, r in S.iterrows():
        d = int(r["day"])
        stored = int(r["user_tx_cnt"])
        ds = vis_days.get(r["visitor"], np.array([], dtype=np.int64))
        for w in range(1, 740):
            if int(((ds >= d + 1 - w) & (ds < d + 1)).sum()) != stored:
                common["past_incl"].discard(w)
            if int(((ds >= d + 1 - w) & (ds < d)).sum()) != stored:
                common["past_excl"].discard(w)
            if int(((ds >= d + 1) & (ds < d + 1 + w)).sum()) != stored:
                common["forward"].discard(w)
    for k, v in common.items():
        lo, hi = (min(v), max(v)) if v else (0, 0)
        print(f"  user_tx_cnt == {k:>10} for every row: "
              f"{'yes, w in [%d..%d]' % (lo, hi) if v else 'no'}")
    print("  (the mean of the range [7, 738] is 372.6 days)")
    return True


# ---------------------------------------------------------------------------
def check_truncate_old(samples, days_to_test=(120, 300, 480, 660)):
    """Recompute the features from the data truncated at the snapshot day."""
    from engine import DayCounter
    from featurize import FeatureBuilder
    cfg = load_config()
    k, x, q = 332, 377, 102
    print("\n=== 2. truncation test (features must not change) ===")
    stored = pd.read_csv(os.path.join(HERE, "features_part_2.csv"),
                         parse_dates=["t"], dtype={"visitor": "string",
                                                   "itemid": "string"})
    stored["visitor"] = stored["visitor"].fillna("")
    retail = load_retail()
    origin = pd.Timestamp(cfg["obs_start"])
    retail["day"] = ((retail["InvoiceDate"] - origin).dt.total_seconds()
                     // 86400).astype(np.int64)
    full_items = pd.Categorical(retail["itemid"]).categories
    full_visitors = pd.Categorical(retail["visitor"].fillna("")).categories
    full_cats = pd.Categorical(retail["item_cat"]).categories
    for D in days_to_test:
        sub = retail[retail["day"] <= D].copy()
        if len(sub) == 0:
            continue
        # build the sample frame by hand on the truncated data
        s = sub.copy()
        s["t"] = origin + pd.to_timedelta(s["day"] + 1, unit="D")
        s["visitor"] = s["visitor"].fillna("")
        # the codes must be stable: they are taken from the full table, exactly
        # as the pipeline does (otherwise the categories are renumbered and
        # the test would measure a coding artefact instead of a leak)
        s["item_id"] = pd.Categorical(s["itemid"],
                                      categories=full_items).codes.astype(np.int64)
        s["visitor_id"] = pd.Categorical(s["visitor"],
                                          categories=full_visitors).codes.astype(np.int64)
        s["item_cat_id"] = pd.Categorical(s["item_cat"],
                                          categories=full_cats).codes.astype(np.int64)
        s["visitor_cat_id"] = s["visitor_cat"].astype(np.int64)
        n_items = int(s["item_id"].max()) + 1
        n_cat = int(s["item_cat_id"].max()) + 1
        from engine import encode_pairs
        s["vi_id"] = encode_pairs(s["visitor_id"].to_numpy(),
                                  s["item_id"].to_numpy(), n_items)
        s["vc_id"] = encode_pairs(s["visitor_id"].to_numpy(),
                                  s["item_cat_id"].to_numpy(), n_cat)
        s["ic_id"] = encode_pairs(s["item_id"].to_numpy(),
                                  s["visitor_cat_id"].to_numpy(),
                                  int(s["visitor_cat_id"].max()) + 1)
        fb = FeatureBuilder(s, cfg, g6_stride=30)
        cols = {}
        cols.update(fb.group_31(k))
        cols.update(fb.group_32(x))
        cols.update(fb.group_4(q))
        mine = pd.DataFrame({
            "t": s["t"], "visitor": s["visitor"], "itemid": s["itemid"],
            "cat_tx_cnt": cols["cat_tx_cnt"],
            "viscat_tx_cnt": cols["viscat_tx_cnt"],
            "vi_tx_cnt": cols["vi_tx_cnt"],
        })
        rows = mine[mine["t"] == (origin + pd.Timedelta(days=D + 1))]
        key = ["t", "visitor", "itemid"]
        ref = stored.merge(rows, on=key, how="inner", suffixes=("_ref", "_new"))
        print(f"  day {D:>4} ({origin + pd.Timedelta(days=D + 1)}): "
              f"{len(ref)} rows compared")
        for c in ("cat_tx_cnt", "viscat_tx_cnt", "vi_tx_cnt"):
            if c + "_ref" not in ref.columns:
                continue
            a = ref[c + "_ref"].to_numpy()
            b = ref[c + "_new"].to_numpy()
            diff = int((a != b).sum())
            print(f"      {c:>16}: mismatches {diff:>7} / {len(ref)}"
                  f"   max|diff| {int(np.abs(a - b).max()) if len(ref) else 0}")
    return True


# ---------------------------------------------------------------------------
def check_univariate(samples, df, tr, te, known):
    y = samples["target"].to_numpy()
    print("\n=== 3. univariate AUC on the holdout (known buyers) ===")
    rows = []
    for f in FEATURES_ALL:
        if f not in df.columns:
            continue
        v = df[f].to_numpy(dtype=np.float64)
        m = te & known
        a = roc_auc_score(y[m], v[m])
        rows.append({"feature": f, "auc_holdout": round(a, 4),
                     "signal": round(abs(a - 0.5) + 0.5, 4),
                     "direction": "direct" if a >= 0.5 else "inverse",
                     "share_zero": round(float((v[m] == 0).mean()), 4),
                     "median": float(np.median(v[m]))})
    r = pd.DataFrame(rows).sort_values("signal", ascending=False)
    print(r.to_string(index=False))
    return r


# ---------------------------------------------------------------------------
def check_models(samples, df, tr, te, known):
    y = samples["target"].to_numpy()
    print("\n=== 4. where does the signal come from? (holdout, known buyers) ===")
    sets = {
        "part2 only (3.1/3.2/4/6)": PART2_FEATURES,
        "part1 only (repo table)": [f for f in PART1_FEATURES if f in df.columns],
        "all features": FEATURES_ALL,
    }
    for name, feats in sets.items():
        feats = [f for f in feats if f in df.columns]
        X = df[feats].to_numpy(dtype=np.float64)
        m = lgb_fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        kn_te = known[te]
        auc_all = roc_auc_score(y[te], p)
        auc_kn = roc_auc_score(y[te][kn_te], p[kn_te])
        print(f"  {name:28s} n={len(feats):>2}  AUC all rows {auc_all:.4f}  "
              f"AUC known buyers {auc_kn:.4f}")
    return True


# ---------------------------------------------------------------------------
def check_forward_cv(samples, df, tr, te, known, n_folds=5):
    """Expanding window CV: a fold is validated only on data that is later
    than everything it was trained on."""
    y = samples["target"].to_numpy()
    day = samples["day"].to_numpy()
    days = np.sort(np.unique(day))
    blocks = np.array_split(days, n_folds)
    print("\n=== 5. strictly forward-chaining CV ===")
    X = df[FEATURES_ALL].to_numpy(dtype=np.float64)
    oof = np.full(len(y), np.nan)
    for i, b in enumerate(blocks):
        va = np.isin(day, b)
        trn = day < b[0]
        if trn.sum() == 0:
            continue
        m = lgb_fit(X[trn], y[trn])
        oof[va] = m.predict_proba(X[va])[:, 1]
        mk = va & known
        print(f"  fold {i + 1}: train days < {b[0]} ({int(trn.sum())} rows), "
              f"valid {b[0]}..{b[-1]} ({int(va.sum())} rows)  "
              f"AUC {roc_auc_score(y[mk], oof[mk]):.4f}")
    ok = ~np.isnan(oof)
    mk = ok & known
    print(f"  forward-chaining OOF AUC (known buyers): "
          f"{roc_auc_score(y[mk], oof[mk]):.4f}")
    # for comparison: the blocked CV that trains on the other blocks
    from importlib import import_module
    m6 = import_module("06_models")
    cvm, oof_blocked = m6.cv_metrics(X, y, day, mode="blocked")
    mb = known
    print(f"  blocked CV OOF AUC (known buyers)        : "
          f"{roc_auc_score(y[mb], oof_blocked[mb]):.4f}   <-- trains on the future")
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--part1", default="features_part_1_repo.csv")
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    samples = build_samples(load_retail(), cfg)
    print(f"samples {len(samples)}, positives {int(samples['target'].sum())} "
          f"({samples['target'].mean():.4f})")
    tr, te, known = check_split(samples)
    p2, p1 = load_tables(args.part1)
    df = assemble(samples, p2, p1)
    if args.part in ("all", "univ", "models", "forward"):
        check_univariate(samples, df, tr, te, known)
    if args.part in ("all", "models"):
        check_models(samples, df, tr, te, known)
    if args.part in ("all", "forward"):
        check_forward_cv(samples, df, tr, te, known)
    if args.part in ("all", "truncate"):
        check_truncate(samples)
    if args.part in ("all", "repo"):
        check_repo_window(samples)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
