"""Modelling dataset: samples, target, feature assembly.

Target (horizon = 7 days): for a sample transaction of (visitor, itemid) on
day d, with the snapshot t = end of day d, the target is 1 when the visitor
buys the same itemid again inside the target window [t, t + 7 days), i.e. on
one of the days d+1 .. d+7.  The current transaction is on day d and therefore
outside the target window (half-open window, same convention as the features).
"""
import os

import numpy as np
import pandas as pd

from config import load_config
from engine import day_index

HERE = os.path.dirname(os.path.abspath(__file__))


def load_retail():
    df = pd.read_csv(os.path.join(HERE, "retail.csv"),
                     dtype={"itemid": "string", "visitor": "string"},
                     parse_dates=["InvoiceDate"])
    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    return df


def build_samples(df, cfg=None):
    """Add day index, snapshot time, keys and the binary target."""
    cfg = cfg or load_config()
    origin = pd.Timestamp(cfg["obs_start"])
    df = df.copy()
    df["day"] = day_index(df["InvoiceDate"], origin)
    df["t"] = origin + pd.to_timedelta(df["day"] + 1, unit="D")
    df["visitor"] = df["visitor"].fillna("")

    # ---- target: repeat purchase of the same item inside [t, t+horizon) ------
    # The transactions without a customer id have no buyer: they cannot show a
    # repeat purchase of the same buyer, so their target is 0.  They must not
    # be merged into one pseudo-visitor either - otherwise all the anonymous
    # transactions of an item look like a "repeat purchase" of one buyer.
    horizon = cfg["horizon"]
    has_visitor = df["visitor"].ne("")
    trip = df.loc[has_visitor, ["visitor", "itemid", "day"]].drop_duplicates()
    trip = trip.sort_values(["visitor", "itemid", "day"], kind="stable")
    next_day = trip.groupby(["visitor", "itemid"], sort=False)["day"].shift(-1)
    trip["next_day"] = next_day
    lookup = trip.set_index(["visitor", "itemid", "day"])["next_day"]
    idx = pd.MultiIndex.from_frame(df[["visitor", "itemid", "day"]])
    nxt = lookup.reindex(idx).to_numpy()
    delta = nxt - df["day"].to_numpy()
    df["target"] = ((delta >= 1) & (delta <= horizon)).astype(np.int8)
    df["target"] = df["target"].astype(np.int8)
    df.attrs["has_visitor"] = has_visitor.to_numpy()
    return df


def temporal_split(df, test_frac=0.2):
    """Split by snapshot time: the last `test_frac` of the snapshots is test."""
    days = np.sort(df["day"].unique())
    cut = days[int(len(days) * (1 - test_frac))]
    return (df["day"] < cut), (df["day"] >= cut)


def load_features_part2(path="features_part_2.csv"):
    f = pd.read_csv(path, parse_dates=["t"])
    return f


def load_features_part1(path="features_part_1.csv"):
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path)
    return f


def build_model_frame(samples, part2, part1=None, feature_cols=None):
    """Join the feature tables onto the sample rows."""
    df = samples.merge(part2, on=["t", "visitor", "itemid"], how="left",
                       suffixes=("", "_p2"))
    if part1 is not None:
        df = df.merge(part1, on=["t", "visitor", "itemid"], how="left",
                      suffixes=("", "_p1"))
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in ("t", "visitor", "itemid", "item_cat",
                                     "visitor_cat", "day", "target",
                                     "InvoiceDate", "Quantity")]
    return df, feature_cols


if __name__ == "__main__":
    cfg = load_config()
    df = build_samples(load_retail(), cfg)
    print("rows:", len(df), "positives:", int(df["target"].sum()),
          f"rate: {df['target'].mean():.4f}")
    print(df["target"].value_counts().to_string())
    tr, te = temporal_split(df)
    print("train rows:", int(tr.sum()), "test rows:", int(te.sum()))
    print("train pos rate: %.4f  test pos rate: %.4f"
          % (df.loc[tr, "target"].mean(), df.loc[te, "target"].mean()))
    print(df.head(5).to_string())
