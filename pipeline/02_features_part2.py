"""Step 2: features of the groups 3.1, 3.2, 4 and 6 -> features_part_2.csv.

Sample grid
-----------
t is a day boundary produced by the pipeline: the snapshot at the end of an
observed day.  W(t, w) = [t-w, t) therefore covers exactly the whole days
[d+1-w, d+1).  One sample row per transaction, carrying its (visitor, itemid).

Groups
------
3.1  key item_cat          cat_tx_cnt(c, t, k)
3.2  key visitor_cat       viscat_tx_cnt(c, t, x)
4    key (visitor, itemid)  vi_tx_cnt(v, i, t, q)
6    key (itemid, itemid)   ii_co_purchase(i, j, t, u)

The relation of group 6 has its own key, so it is written to a separate file
features_part_2_group6.csv; features_part_2.csv additionally carries the
row-grain projection ii_co_purchase_cnt of that relation.
"""
import argparse
import os

import pandas as pd

from config import load_config
from featurize import FeatureBuilder, prepare_samples

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--u", type=int, default=None)
    ap.add_argument("--out", default="features_part_2.csv")
    ap.add_argument("--out-g6", default="features_part_2_group6.csv")
    ap.add_argument("--out-g6-pairs", default="features_part_2_group6_pairs.csv.gz")
    ap.add_argument("--g6-stride", type=int, default=30)
    ap.add_argument("--g6-top-items", type=int, default=300,
                    help="the pair relation is materialised for the N most "
                         "frequent items only (it is combinatorial otherwise)")
    ap.add_argument("--no-pair-table", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    k = args.k if args.k is not None else cfg["param_mean"]
    x = args.x if args.x is not None else cfg["param_mean"]
    q = args.q if args.q is not None else cfg["param_mean"]
    u = args.u if args.u is not None else cfg["param_mean"]

    samples = prepare_samples(cfg=cfg)
    fb = FeatureBuilder(samples, cfg, g6_stride=args.g6_stride)
    print(f"rows={len(samples)}  params k={k} x={x} q={q} u={u}")

    cols = {}
    cols.update(fb.group_31(k))
    cols.update(fb.group_32(x))
    cols.update(fb.group_4(q))
    cols.update(fb.group_6(u))

    out = pd.DataFrame({
        "t": samples["t"],
        "visitor": samples["visitor"],
        "itemid": samples["itemid"],
        "item_cat": samples["item_cat"],
        "visitor_cat": samples["visitor_cat"],
        "cat_tx_cnt": cols["cat_tx_cnt"],
        "viscat_tx_cnt": cols["viscat_tx_cnt"],
        "vi_tx_cnt": cols["vi_tx_cnt"],
        "ii_co_purchase_cnt": cols["ii_co_purchase_cnt"],
    })
    out.to_csv(os.path.join(HERE, args.out), index=False, encoding="utf-8",
               lineterminator="\r\n")
    print("wrote", args.out, out.shape)

    if not args.no_pair_table:
        # per item projection of the group 6 relation (complete, small)
        g6i = fb.group6_item_table(u)
        g6i.to_csv(os.path.join(HERE, args.out_g6), index=False, encoding="utf-8",
                   lineterminator="\r\n")
        size = os.path.getsize(os.path.join(HERE, args.out_g6)) / 1e6
        print(f"wrote {args.out_g6} {g6i.shape}  {size:.1f} MB")

        # the pair relation itself, restricted to the most frequent items
        top = fb.top_items(args.g6_top_items)
        pairs = fb.group6_pair_table(u, top_items=top)
        gz = os.path.join(HERE, args.out_g6_pairs)
        pairs.to_csv(gz, index=False, encoding="utf-8", compression="gzip", lineterminator="\r\n")
        size = os.path.getsize(gz) / 1e6
        print(f"wrote {args.out_g6_pairs} {pairs.shape}  positives="
              f"{int(pairs['ii_co_purchase'].sum())}  {size:.1f} MB "
              f"(top {args.g6_top_items} items, stride {args.g6_stride} days)")

    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
