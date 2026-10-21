"""Fallback generator of the feature groups 1, 2 and 5.1 (and 5.2 partially).

Used only when features_part_1.csv is not present in the repository, so that
the final models can still be trained on a complete feature set.

Group 1  key visitor        user_tx_cnt(v,t,n), user_avg_active_day_gap(v,t,n)
Group 2  key itemid         item_tx_cnt(i,t,m), item_avg_active_day_gap(i,t,m),
                             item_repeat_purchase_rate(i,t,m)
Group 5.1 key (visitor,item_cat)  vc_tx_cnt(v,c,t,s)
Group 5.2 key (itemid,visitor_cat) ic_tx_cnt(i,c,t,z)
        (ic_view_to_tx_conv is not reproducible here: the data set contains
         purchases only, there is no view log)
"""
import argparse
import os

import numpy as np
import pandas as pd

from config import load_config
from engine import DayCounter, encode_pairs

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--s", type=int, default=None)
    ap.add_argument("--z", type=int, default=None)
    ap.add_argument("--out", default="features_part_1_local.csv")
    args = ap.parse_args()

    cfg = load_config()
    n = args.n or cfg["param_mean"]
    m = args.m or cfg["param_mean"]
    s = args.s or cfg["param_mean"]
    z = args.z or cfg["param_mean"]
    n_days = cfg["n_days"]

    df = pd.read_csv(os.path.join(HERE, "retail.csv"),
                     dtype={"itemid": "string", "visitor": "string"},
                     parse_dates=["InvoiceDate"])
    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    origin = pd.Timestamp(cfg["obs_start"])
    df["day"] = ((df["InvoiceDate"] - origin).dt.total_seconds() // 86400).astype(np.int64)
    df["t"] = origin + pd.to_timedelta(df["day"] + 1, unit="D")
    df["visitor"] = df["visitor"].fillna("")

    item_code = pd.Categorical(df["itemid"]).codes.astype(np.int64)
    df["item_id"] = item_code
    vis_code = pd.Categorical(df["visitor"]).codes.astype(np.int64)
    df["visitor_id"] = vis_code
    cat_code = pd.Categorical(df["item_cat"]).codes.astype(np.int64)
    df["item_cat_id"] = cat_code
    vcat = df["visitor_cat"].astype(np.int64)
    n_items = int(item_code.max()) + 1
    n_vis = int(vis_code.max()) + 1
    n_cat = int(cat_code.max()) + 1
    n_vcat = int(vcat.max()) + 1

    day = df["day"].to_numpy()
    day_end = day + 1

    # ---- group 1 -----------------------------------------------------------
    dc_v = DayCounter(vis_code, day, n_vis, n_days)
    user_tx_cnt = dc_v.window_count(vis_code, day_end, n)
    user_avg_active_day_gap = dc_v.active_day_gap(vis_code, day_end, n)

    # ---- group 2 -----------------------------------------------------------
    dc_i = DayCounter(item_code, day, n_items, n_days)
    item_tx_cnt = dc_i.window_count(item_code, day_end, m)
    item_avg_active_day_gap = dc_i.active_day_gap(item_code, day_end, m)

    # item_repeat_purchase_rate: users who bought the item on >= 2 distinct
    # days inside the window.  Key of the intermediate index is (item, user).
    iu_raw = encode_pairs(item_code, vis_code, n_vis)
    iu, iu_inv = np.unique(iu_raw, return_inverse=True)      # dense pair ids
    n_iu = len(iu)
    dc_iu = DayCounter(iu_inv, day, n_iu, n_days)
    # unique (pair, snapshot) combinations
    span = n_days + 1                      # day_end can reach n_days
    ps = iu_inv * span + day_end
    ps_u, ps_inv = np.unique(ps, return_inverse=True)
    ps_day_end = (ps_u % span).astype(np.int64)
    ps_pair = (ps_u // span).astype(np.int64)
    days_of_pair = dc_iu.active_days(ps_pair, ps_day_end, m)   # distinct days
    valid = days_of_pair >= 2
    ps_item = iu[ps_pair] // n_vis                              # item of the pair
    comb = ps_item[valid] * span + ps_day_end[valid]
    uniq_comb, counts = np.unique(comb, return_counts=True)
    # map every row back to the count of its (item, snapshot)
    row_comb = item_code * span + day_end
    pos = np.searchsorted(uniq_comb, row_comb, "left")
    item_repeat_purchase_rate = np.zeros(len(df), dtype=np.int64)
    has = pos < len(uniq_comb)
    item_repeat_purchase_rate[has] = counts[pos[has]]

    # ---- group 5.1 ---------------------------------------------------------
    vc = encode_pairs(vis_code, cat_code, n_cat)
    n_vc = int(vc.max()) + 1
    dc_vc = DayCounter(vc, day, n_vc, n_days)
    vc_tx_cnt = dc_vc.window_count(vc, day_end, s)

    # ---- group 5.2 ---------------------------------------------------------
    ic = encode_pairs(item_code, vcat, n_vcat)
    n_ic = int(ic.max()) + 1
    dc_ic = DayCounter(ic, day, n_ic, n_days)
    ic_tx_cnt = dc_ic.window_count(ic, day_end, z)

    out = pd.DataFrame({
        "t": df["t"],
        "visitor": df["visitor"],
        "itemid": df["itemid"],
        "user_tx_cnt": user_tx_cnt,
        "user_avg_active_day_gap": user_avg_active_day_gap,
        "item_tx_cnt": item_tx_cnt,
        "item_avg_active_day_gap": item_avg_active_day_gap,
        "item_repeat_purchase_rate": item_repeat_purchase_rate,
        "vc_tx_cnt": vc_tx_cnt,
        "ic_tx_cnt": ic_tx_cnt,
    })
    out.to_csv(os.path.join(HERE, args.out), index=False, encoding="utf-8",
               lineterminator="\r\n")
    print("wrote", args.out, out.shape)
    print(out.describe().T.to_string())


if __name__ == "__main__":
    main()
