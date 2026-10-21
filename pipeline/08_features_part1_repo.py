"""Bring the repository table features_part_1 (feature_1.csv) to the pipeline
conventions.

The repository table stores `t` as the unix timestamp of the transaction
itself, while the pipeline uses `t` = the snapshot, i.e. midnight at the end of
the observed day.  This script converts the timestamp into the snapshot of its
day, removes the duplicated keys and writes the result in the repository
format (CRLF, UTF-8 without BOM).

Rows that are not covered by the repository table (transactions without a
customer id, items that only anonymous customers bought) keep the value 0,
which is the convention of the task for an empty window.
"""
import os

import numpy as np
import pandas as pd

from config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def main(src="feature_1_downloaded.csv", out="features_part_1_repo.csv"):
    cfg = load_config()
    origin = pd.Timestamp(cfg["obs_start"])

    df = pd.read_csv(os.path.join(HERE, src), dtype={"visitor": "string",
                                                      "itemid": "string"})
    print(f"{src}: {len(df)} rows, {len(df.columns)} columns")

    # unix timestamp -> the snapshot of the day of the transaction
    ts = pd.to_datetime(pd.Series(df["t"].to_numpy()), unit="s")
    df["t"] = (ts.dt.normalize() + pd.Timedelta(days=1)).to_numpy()
    df["visitor"] = df["visitor"].fillna("")

    key = ["t", "visitor", "itemid"]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    print(f"duplicated (t, visitor, itemid) rows removed: {before - len(df)}")
    print(f"t range: {df['t'].min()} .. {df['t'].max()}")

    df.to_csv(os.path.join(HERE, out), index=False, encoding="utf-8",
              lineterminator="\r\n")
    print("wrote", out, df.shape)

    # ---- check the coverage against the pipeline samples -------------------
    import sys
    sys.path.insert(0, HERE)
    from dataset import build_samples, load_retail
    samples = build_samples(load_retail(), cfg)
    s = samples[["t", "visitor", "itemid"]].copy()
    s["visitor"] = s["visitor"].fillna("")
    merged = s.merge(df[key + ["user_tx_cnt"]], on=key, how="left",
                     validate="many_to_one")
    hit = merged["user_tx_cnt"].notna()
    print(f"sample rows: {len(s)}  matched: {int(hit.sum())} "
          f"({hit.mean():.4%})  unmatched: {int((~hit).sum())}")
    matched_with_visitor = hit & (merged["visitor"] != "")
    print(f"  of the rows with a known visitor: {int(matched_with_visitor.sum())} "
          f"of {int((merged['visitor'] != '').sum())}")


if __name__ == "__main__":
    main()
