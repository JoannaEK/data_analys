"""Feature construction for every group, at arbitrary window parameters.

One instance holds the day-indexed counters of all keys; the window length is
passed per call, so any parameter combination can be evaluated cheaply.
"""
import os

import numpy as np
import pandas as pd

from config import load_config
from engine import DayCounter, all_pairs_within_groups, encode_pairs

HERE = os.path.dirname(os.path.abspath(__file__))


def prepare_samples(retail_csv="retail.csv", cfg=None):
    """Read the edited retail table and attach the keys used by the features."""
    cfg = cfg or load_config()
    origin = pd.Timestamp(cfg["obs_start"])
    df = pd.read_csv(os.path.join(HERE, retail_csv),
                     dtype={"itemid": "string", "visitor": "string"},
                     parse_dates=["InvoiceDate"])
    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    df["day"] = ((df["InvoiceDate"] - origin).dt.total_seconds() // 86400).astype(np.int64)
    df["t"] = origin + pd.to_timedelta(df["day"] + 1, unit="D")
    df["visitor"] = df["visitor"].fillna("")

    df["item_id"] = pd.Categorical(df["itemid"]).codes.astype(np.int64)
    df["visitor_id"] = pd.Categorical(df["visitor"]).codes.astype(np.int64)
    df["item_cat_id"] = pd.Categorical(df["item_cat"]).codes.astype(np.int64)
    df["visitor_cat_id"] = df["visitor_cat"].astype(np.int64)

    n_items = int(df["item_id"].max()) + 1
    n_vis = int(df["visitor_id"].max()) + 1
    n_cat = int(df["item_cat_id"].max()) + 1
    df["vi_id"] = encode_pairs(df["visitor_id"].to_numpy(), df["item_id"].to_numpy(), n_items)
    df["vc_id"] = encode_pairs(df["visitor_id"].to_numpy(), df["item_cat_id"].to_numpy(), n_cat)
    df["ic_id"] = encode_pairs(df["item_id"].to_numpy(), df["visitor_cat_id"].to_numpy(),
                               int(df["visitor_cat_id"].max()) + 1)
    df.attrs["n_items"] = n_items
    df.attrs["n_vis"] = n_vis
    df.attrs["n_cat"] = n_cat
    return df


class FeatureBuilder:
    """All feature groups, evaluated at any window length."""

    def __init__(self, df, cfg=None, g6_stride=30):
        self.df = df
        self.cfg = cfg or load_config()
        self.n_days = self.cfg["n_days"]
        self.g6_stride = g6_stride
        day = df["day"].to_numpy()
        self.day = day
        self.day_end = day + 1

        self.item_key = df["item_id"].to_numpy()
        self.itemid_values = pd.Categorical(df["itemid"]).categories.to_numpy()
        self.visitor_key = df["visitor_id"].to_numpy()
        self.cat_key = df["item_cat_id"].to_numpy()
        self.vcat_key = df["visitor_cat_id"].to_numpy()
        self.vi_key = df["vi_id"].to_numpy()
        self.vc_key = df["vc_id"].to_numpy()
        self.ic_key = df["ic_id"].to_numpy()

        self.n_items = int(self.item_key.max()) + 1
        self.n_vis = int(self.visitor_key.max()) + 1
        self.n_cat = int(self.cat_key.max()) + 1
        self.n_vcat = int(self.vcat_key.max()) + 1
        self.n_vi = int(self.vi_key.max()) + 1
        self.n_vc = int(self.vc_key.max()) + 1
        self.n_ic = int(self.ic_key.max()) + 1

        # The transactions without a customer id have no visitor.  They are
        # not merged into one pseudo-visitor: the visitor-keyed counters are
        # built on the identified buyers only, so every visitor-keyed feature
        # of such a row is 0 ("no visitor -> no visitor level count").
        # The category-keyed features (3.1, 3.2, 5.2) do count them: they are
        # transactions of a category.
        if "visitor" in df.columns:
            self.known = df["visitor"].ne("").to_numpy()
        else:
            self.known = np.ones(len(df), dtype=bool)
        kn = self.known

        self.dc_vis = DayCounter(self.visitor_key[kn], day[kn], self.n_vis,
                                 self.n_days)
        self.dc_item = DayCounter(self.item_key, day, self.n_items, self.n_days)
        self.dc_cat = DayCounter(self.cat_key, day, self.n_cat, self.n_days)
        self.dc_vcat = DayCounter(self.vcat_key, day, self.n_vcat, self.n_days)
        self.dc_vi = DayCounter(self.vi_key[kn], day[kn], self.n_vi, self.n_days)
        self.dc_vc = DayCounter(self.vc_key[kn], day[kn], self.n_vc, self.n_days)
        self.dc_ic = DayCounter(self.ic_key, day, self.n_ic, self.n_days)

        # (item, user) pairs for item_repeat_purchase_rate
        iu_raw = encode_pairs(self.item_key, self.visitor_key, self.n_vis)
        iu_known = np.unique(iu_raw[kn])
        # the last slot is reserved for the rows without a buyer: it is not a
        # valid (item, visitor) pair, so it is never present in the counter
        self.iu = np.append(iu_known, int(iu_known.max()) + 1)
        self.iu_inv = np.full(len(self.df), len(self.iu) - 1, dtype=np.int64)
        pos = np.searchsorted(self.iu, iu_raw, side="left")
        self.iu_inv[kn] = np.clip(pos[kn], 0, len(self.iu) - 2)
        self.dc_iu = DayCounter(self.iu_inv[kn], day[kn], len(self.iu), self.n_days)

        self.g6_snaps = list(range(0, self.n_days, g6_stride))
        self._g6_cache = {}

    # ---------------- groups 1, 2, 3.1, 3.2, 4, 5.1, 5.2 -------------------
    def group_1(self, n):
        return {
            "user_tx_cnt": self.dc_vis.window_count(self.visitor_key, self.day_end, n),
            "user_avg_active_day_gap": self.dc_vis.active_day_gap(self.visitor_key, self.day_end, n),
        }

    def group_2(self, m):
        days_of_pair = self.dc_iu.active_days(self.iu_inv, self.day_end, m)
        span = self.n_days + 1
        ps = self.iu_inv * span + self.day_end
        ps_u = np.unique(ps)
        ps_day_end = (ps_u % span).astype(np.int64)
        ps_pair = (ps_u // span).astype(np.int64)
        valid = self.dc_iu.active_days(ps_pair, ps_day_end, m) >= 2
        ps_item = self.iu[ps_pair] // self.n_vis
        comb = ps_item[valid] * span + ps_day_end[valid]
        uniq_comb, counts = np.unique(comb, return_counts=True)
        row_comb = self.item_key * span + self.day_end
        pos = np.searchsorted(uniq_comb, row_comb, "left")
        rate = np.zeros(len(self.df), dtype=np.int64)
        has = pos < len(uniq_comb)
        rate[has] = counts[pos[has]]
        return {
            "item_tx_cnt": self.dc_item.window_count(self.item_key, self.day_end, m),
            "item_avg_active_day_gap": self.dc_item.active_day_gap(self.item_key, self.day_end, m),
            "item_repeat_purchase_rate": rate,
        }

    def group_31(self, k):
        return {"cat_tx_cnt": self.dc_cat.window_count(self.cat_key, self.day_end, k)}

    def group_32(self, x):
        return {"viscat_tx_cnt": self.dc_vcat.window_count(self.vcat_key, self.day_end, x)}

    def group_4(self, q):
        return {"vi_tx_cnt": self.dc_vi.window_count(self.vi_key, self.day_end, q)}

    def group_51(self, s):
        return {"vc_tx_cnt": self.dc_vc.window_count(self.vc_key, self.day_end, s)}

    def group_52(self, z):
        return {"ic_tx_cnt": self.dc_ic.window_count(self.ic_key, self.day_end, z)}

    # ---------------- group 6 ----------------------------------------------
    def group6_degree(self, u):
        """(n_snapshots, n_items): for every item i the number of distinct
        items j with ii_co_purchase(i, j, t, u) = 1."""
        if u in self._g6_cache:
            return self._g6_cache[u]
        deg = np.zeros((len(self.g6_snaps), self.n_items), dtype=np.int32)
        for si, d in enumerate(self.g6_snaps):
            lo, hi = d + 1 - u, d + 1
            m = (self.day >= lo) & (self.day < hi) & self.known
            key = encode_pairs(self.visitor_key[m], self.item_key[m], self.n_items)
            key = np.unique(key)
            if len(key) < 2:
                continue
            uv, ui = key // self.n_items, key % self.n_items
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

    def group_6(self, u):
        """Row-grain projection of the group 6 relation."""
        deg = self.group6_degree(u)
        snap_end = np.asarray(self.g6_snaps, dtype=np.int64) + 1
        pos = np.searchsorted(snap_end, self.day_end, side="right") - 1
        out = np.zeros(len(self.df), dtype=np.int32)
        ok = pos >= 0
        out[ok] = deg[pos[ok], self.item_key[ok]]
        return {"ii_co_purchase_cnt": out}

    def top_items(self, n_top):
        """Codes of the n_top most frequent items (whole observation time)."""
        counts = np.bincount(self.item_key, minlength=self.n_items)
        return np.argsort(-counts, kind="stable")[:n_top]

    def group6_pair_table(self, u, top_items=None):
        """The full group 6 relation: one row per (t, itemid_i, itemid_j)."""
        origin = pd.Timestamp(self.cfg["obs_start"])
        frames = []
        keep_set = None
        if top_items is not None:
            keep_set = np.zeros(self.n_items, dtype=bool)
            keep_set[np.asarray(top_items)] = True
        for d in self.g6_snaps:
            lo, hi = d + 1 - u, d + 1
            m = (self.day >= lo) & (self.day < hi) & self.known
            key = np.unique(encode_pairs(self.visitor_key[m], self.item_key[m], self.n_items))
            if len(key) < 2:
                continue
            uv, ui = key // self.n_items, key % self.n_items
            a, b = all_pairs_within_groups(uv, ui)
            if len(a) == 0:
                continue
            pair = encode_pairs(np.minimum(a, b), np.maximum(a, b), self.n_items)
            uniq, cnt = np.unique(pair, return_counts=True)
            pa = uniq // self.n_items
            pb = uniq % self.n_items
            if keep_set is not None:
                sel = keep_set[pa] & keep_set[pb]
                pa, pb, cnt = pa[sel], pb[sel], cnt[sel]
            frames.append(pd.DataFrame({
                "t": origin + pd.Timedelta(days=d + 1),
                "itemid_i": self.itemid_values[pa],
                "itemid_j": self.itemid_values[pb],
                "ii_co_purchase": (cnt >= 2).astype(np.int64),
            }))
        if not frames:
            return pd.DataFrame(columns=["t", "itemid_i", "itemid_j", "ii_co_purchase"])
        return pd.concat(frames, ignore_index=True)

    def group6_item_table(self, u):
        """Per-item projection of the group 6 relation.

        ii_co_purchase_cnt(i, t, u) = number of distinct items j with
        ii_co_purchase(i, j, t, u) = 1.
        """
        deg = self.group6_degree(u)
        origin = pd.Timestamp(self.cfg["obs_start"])
        ts = [origin + pd.Timedelta(days=d + 1) for d in self.g6_snaps]
        codes = np.repeat(np.arange(self.n_items), len(self.g6_snaps))
        values = deg.T.reshape(-1)
        return pd.DataFrame({
            "t": np.repeat(ts, self.n_items),
            "itemid_i": self.itemid_values[codes],
            "ii_co_purchase_cnt": values.astype(np.int64),
        })

    # ---------------- generic access ---------------------------------------
    GROUP_FUNCS = {
        "n": "group_1", "m": "group_2", "k": "group_31", "x": "group_32",
        "q": "group_4", "s": "group_51", "z": "group_52", "u": "group_6",
    }

    def build(self, params, only=None):
        """params: dict with the window length per parameter name."""
        out = {}
        for pname, fn in self.GROUP_FUNCS.items():
            if only is not None and pname not in only:
                continue
            w = params[pname]
            out.update(getattr(self, fn)(w))
        return out
