"""Experiment 1: adaptive pattern search for the window parameters k, x, q, u.

Search space : each parameter in [7, Y], Y = whole observation time in days.
Start        : every parameter at the mean of its range.
Step         : +/-80, then 40, 20, 10, 5 (halved after both directions of a
               coordinate are exhausted); step < 5 -> converged.
Searched     : k, x, q, u only; n, m, s, z stay at the mean.
Objective    : validation AUC of a LightGBM model over the feature groups
               3.1, 3.2, 4 and 6 (the groups whose windows k, x, q, u define).
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import load_config
from dataset import build_samples, load_retail
from engine import DayCounter, all_pairs_within_groups, encode_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------
class FeatureBuilder:
    """Builds the row-grain features of groups 3.1, 3.2, 4 and 6."""

    def __init__(self, df, cfg, g6_stride=30):
        self.df = df
        self.cfg = cfg
        self.g6_stride = g6_stride
        self.n_days = cfg["n_days"]
        self.day_end = df["day"].to_numpy() + 1

        self.cat_key = df["item_cat_id"].to_numpy()
        self.vcat_key = df["visitor_cat_id"].to_numpy()
        self.vi_key = df["vi_id"].to_numpy()
        self.item_key = df["item_id"].to_numpy()
        self.n_cat = int(self.cat_key.max()) + 1
        self.n_vcat = int(self.vcat_key.max()) + 1
        self.n_vi = int(self.vi_key.max()) + 1
        self.n_items = int(self.item_key.max()) + 1

        days = df["day"].to_numpy()
        self.dc_cat = DayCounter(self.cat_key, days, self.n_cat, self.n_days)
        self.dc_vcat = DayCounter(self.vcat_key, days, self.n_vcat, self.n_days)
        self.dc_vi = DayCounter(self.vi_key, days, self.n_vi, self.n_days)

        # group 6 snapshot grid
        self.g6_snaps = list(range(0, self.n_days, g6_stride))
        self._g6_cache = {}

    # ---- groups 3.1 / 3.2 / 4 (cheap, cached per parameter) ----------------
    def group_counts(self, name, window):
        if name == "cat_tx_cnt":
            return self.dc_cat.window_count(self.cat_key, self.day_end, window)
        if name == "viscat_tx_cnt":
            return self.dc_vcat.window_count(self.vcat_key, self.day_end, window)
        if name == "vi_tx_cnt":
            return self.dc_vi.window_count(self.vi_key, self.day_end, window)
        raise KeyError(name)

    # ---- group 6 -----------------------------------------------------------
    def group6_degree(self, u):
        """(n_snapshots, n_items) degree matrix of the co-purchase relation.

        degree(t, i) = number of distinct items j such that ii_co_purchase(i,j,t,u)=1
        """
        if u in self._g6_cache:
            return self._g6_cache[u]
        deg = np.zeros((len(self.g6_snaps), self.n_items), dtype=np.int32)
        day = self.df["day"].to_numpy()
        vis = self.df["visitor_id"].to_numpy()
        it = self.df["item_id"].to_numpy()
        for si, d in enumerate(self.g6_snaps):
            lo, hi = d + 1 - u, d + 1
            m = (day >= lo) & (day < hi)
            sub_v = vis[m]
            sub_i = it[m]
            # unique (visitor,item) inside the window
            key = encode_pairs(sub_v, sub_i, self.n_items)
            key = np.unique(key)
            uv = key // self.n_items
            ui = key % self.n_items
            if len(uv) < 2:
                continue
            a, b = all_pairs_within_groups(uv, ui)
            if len(a) == 0:
                continue
            pair = encode_pairs(np.minimum(a, b), np.maximum(a, b), self.n_items)
            uniq, cnt = np.unique(pair, return_counts=True)
            keep = cnt >= 2
            if not keep.any():
                continue
            pa = (uniq // self.n_items)[keep]
            pb = (uniq % self.n_items)[keep]
            deg[si] = (np.bincount(pa, minlength=self.n_items)
                       + np.bincount(pb, minlength=self.n_items))
        self._g6_cache[u] = deg
        return deg

    def ii_co_purchase_cnt(self, u):
        """Row-grain projection of the group 6 relation (as-of the last
        group-6 snapshot not later than the row's own snapshot)."""
        deg = self.group6_degree(u)
        snap_end = np.asarray(self.g6_snaps, dtype=np.int64) + 1
        pos = np.searchsorted(snap_end, self.day_end, side="right") - 1
        out = np.zeros(len(self.df), dtype=np.int32)
        ok = pos >= 0
        out[ok] = deg[pos[ok], self.item_key[ok]]
        return out


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
class Objective:
    def __init__(self, fb, samples, params, train_mask, valid_mask, cache_size=6):
        self.fb = fb
        self.samples = samples
        self.params = params            # dict of fixed params (n, m, s, z)
        self.train_mask = train_mask
        self.valid_mask = valid_mask
        self.cache = {}
        self.cache_size = cache_size
        self.y = samples["target"].to_numpy()
        self.calls = 0

    def _cached(self, name, window):
        key = (name, window)
        if key not in self.cache:
            if len(self.cache) > self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = self.fb.group_counts(name, window)
        return self.cache[key]

    def features(self, k, x, q, u):
        cols = [
            self._cached("cat_tx_cnt", k),
            self._cached("viscat_tx_cnt", x),
            self._cached("vi_tx_cnt", q),
            self.fb.ii_co_purchase_cnt(u),
        ]
        return np.column_stack(cols)

    def __call__(self, point):
        self.calls += 1
        t0 = time.time()
        k, x, q, u = point
        X = self.features(k, x, q, u)
        model = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.08, num_leaves=63,
            min_child_samples=100, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED,
            n_jobs=-1, verbose=-1,
        )
        model.fit(X[self.train_mask], self.y[self.train_mask])
        p = model.predict_proba(X[self.valid_mask])[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(self.y[self.valid_mask], p)
        self.last_time = time.time() - t0
        return auc


# --------------------------------------------------------------------------
# adaptive pattern search
# --------------------------------------------------------------------------
def pattern_search(objective, names, start, lo, hi, steps=(80, 40, 20, 10, 5),
                   min_step=5):
    """Classic compass/pattern search with a globally shrinking step."""
    point = dict(start)
    best = objective([point[n] for n in names])
    history = [{"point": dict(point), "auc": best, "step": steps[0]}]
    print(f"start {point} auc={best:.5f}", flush=True)

    for step in steps:
        improved = True
        while improved:
            improved = False
            for n in names:
                for direction in (+1, -1):
                    cand = dict(point)
                    val = point[n] + direction * step
                    if not (lo <= val <= hi):
                        continue
                    cand[n] = val
                    auc = objective([cand[k2] for k2 in names])
                    history.append({"point": dict(cand), "auc": auc, "step": step})
                    print(f"  step {step} {n}{'+' if direction>0 else '-'} -> {cand} auc={auc:.5f}",
                          flush=True)
                    if auc > best:
                        best = auc
                        point = cand
                        improved = True
                        break
                if improved:
                    break
    return point, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g6-stride", type=int, default=30)
    ap.add_argument("--out", default="1exp_opt_2.csv")
    ap.add_argument("--history", default="1exp_opt_2_history.json")
    args = ap.parse_args()

    cfg = load_config()
    lo, hi = cfg["param_min"], cfg["param_max"]
    mean = cfg["param_mean"]
    print(f"Y={cfg['Y']}  range=[{lo}, {hi}]  mean={mean}")

    samples = build_samples(load_retail(), cfg)
    samples["item_cat_id"] = pd.Categorical(samples["item_cat"]).codes.astype(np.int64)
    samples["visitor_cat_id"] = samples["visitor_cat"].astype(np.int64)
    items = pd.Categorical(samples["itemid"]).codes.astype(np.int64)
    samples["item_id"] = items
    samples["visitor_id"] = pd.Categorical(samples["visitor"]).codes.astype(np.int64)
    samples["vi_id"] = encode_pairs(samples["visitor_id"].to_numpy(), items, int(items.max()) + 1)

    days = np.sort(samples["day"].unique())
    cut_tr = days[int(len(days) * 0.70)]
    cut_va = days[int(len(days) * 0.85)]
    tr = (samples["day"] < cut_tr).to_numpy()
    va = ((samples["day"] >= cut_tr) & (samples["day"] < cut_va)).to_numpy()
    print(f"train rows {tr.sum()}  valid rows {va.sum()}")

    fb = FeatureBuilder(samples, cfg, g6_stride=args.g6_stride)
    obj = Objective(fb, samples, {}, tr, va)

    names = ["k", "x", "q", "u"]
    start = {n: mean for n in names}
    t0 = time.time()
    point, best, history = pattern_search(obj, names, start, lo, hi)
    print(f"\nBEST auc={best:.5f} point={point}  ({time.time()-t0:.0f}s, {obj.calls} evaluations)")

    pd.DataFrame([point]).to_csv(args.out, index=False, lineterminator="\r\n")
    with open(args.history, "w", encoding="utf-8") as f:
        json.dump({"best_auc": best, "point": point, "history": history}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
