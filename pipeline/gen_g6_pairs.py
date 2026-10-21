"""Regenerate the group 6 pair relation with a wider item scope.

The full relation ii_co_purchase(i, j, t, u) is combinatorial (about 7.4 M
candidate pairs per snapshot for the whole catalogue, more than 200 M rows on
the monthly snapshot grid), so it cannot be uploaded as is.  The relation is
materialised for the most frequent items, which carry the bulk of the
transactions.
"""
import argparse
import os

from config import load_config
from featurize import FeatureBuilder, prepare_samples

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=int, default=None)
    ap.add_argument("--g6-stride", type=int, default=30)
    ap.add_argument("--g6-top-items", type=int, default=1000)
    ap.add_argument("--out", default="features_part_2_group6_pairs.csv.gz")
    args = ap.parse_args()

    cfg = load_config()
    u = args.u if args.u is not None else cfg["param_mean"]
    samples = prepare_samples(cfg=cfg)
    fb = FeatureBuilder(samples, cfg, g6_stride=args.g6_stride)
    top = fb.top_items(args.g6_top_items)
    pairs = fb.group6_pair_table(u, top_items=top)
    path = os.path.join(HERE, args.out)
    pairs.to_csv(path, index=False, encoding="utf-8", compression="gzip", lineterminator="\r\n")
    size = os.path.getsize(path) / 1e6
    print(f"wrote {args.out} {pairs.shape}  positives="
          f"{int(pairs['ii_co_purchase'].sum())}  {size:.1f} MB "
          f"(top {args.g6_top_items} items, stride {args.g6_stride} days, u={u})")


if __name__ == "__main__":
    main()
