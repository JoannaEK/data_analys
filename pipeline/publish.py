"""Upload the pipeline artefacts to the repository.

The contents API of GitHub refuses a file above ~42 MB, so every table that is
bigger is uploaded as a gzip copy (<name>.csv.gz); the content is identical,
only the transport is compressed.

features_part_2.csv        - groups 3.1, 3.2, 4 (row grain) + the row-grain
                             projection of group 6
features_part_2_group6.csv - the per item projection of the group 6 relation
features_part_2_group6_pairs.csv.gz
                           - the group 6 relation itself, key (itemid_i, itemid_j)
1exp_opt_2.csv             - optimal window parameters k, x, q, u
"""
import gzip
import os
import shutil
import sys

import gh_api

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_MB = 40.0          # the contents API refuses a file above ~42 MB

TARGETS = [
    ("features_part_2.csv", "features_part_2.csv: feature groups 3.1, 3.2, 4 and the group 6 projection"),
    ("1exp_opt_2.csv", "1exp_opt_2.csv: optimal window parameters k, x, q, u (experiment 1, adaptive pattern search)"),
    ("1exp_opt_2_history.json", "1exp_opt_2_history.json: full search history of experiment 1"),
    ("features_part_2_group6.csv", "features_part_2_group6.csv: feature group 6, ii_co_purchase_cnt(i, t, u)"),
    ("features_part_2_group6_pairs.csv.gz", "features_part_2_group6_pairs.csv.gz: feature group 6, the pair relation ii_co_purchase(i, j, t, u)"),
    ("retail.csv", "retail.csv: the edited retail table (stock codes, itemid, visitor, visitor_cat, item_cat_1..10)"),
    ("retail_meta.json", "retail_meta.json: metadata of the edited retail table (item_cat bins, visitor_cat map)"),
    ("features_part_1_local.csv", "features_part_1_local.csv: locally generated fallback for the feature groups 1, 2, 5.1, 5.2 (the start windows)"),
    ("features_part_1_local_opt.csv", "features_part_1_local_opt.csv: the feature groups 1, 2, 5.1, 5.2 recomputed locally with the optimal windows of 1exp_opt_1.csv"),
    ("2exp_opt.csv", "2exp_opt.csv: the window parameters of experiment 2 (per feature)"),
    ("2exp_models_comparison.csv", "2exp_models_comparison.csv: the metrics of the models of experiment 2"),
    ("2exp_search_history.json", "2exp_search_history.json: the search history of experiment 2"),
    ("leak_audit.txt", "leak_audit.txt: the leakage audit - the split, the window of every feature, the univariate AUC of every feature, the forward-chaining CV"),
    ("models_comparison.csv", "models_comparison.csv: metrics of the final models"),
    ("model_prosto.csv", "model_prosto.csv: features of every model, the window parameter of every feature, the source of the value and the metrics of the model"),
    ("models_features.csv", "models_features.csv: features and window parameters used by every model"),
    ("models_feature_importance.csv", "models_feature_importance.csv: feature importance of every model"),
    ("config.json", "config.json: pipeline conventions (observation window, Y, parameter range, horizon)"),
    ("REPORT.md", "REPORT.md: the report of the pipeline: features, experiments, models and metrics"),
]


def gzip_copy(path):
    """Create <path>.gz next to the file and return its path."""
    gz = path + ".gz"
    if os.path.exists(gz) and os.path.getmtime(gz) >= os.path.getmtime(path):
        return gz
    with open(path, "rb") as fin, gzip.open(gz, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    return gz


def publish(only=None):
    for local, message in TARGETS:
        if only and local not in only:
            continue
        path = os.path.join(HERE, local)
        if not os.path.exists(path):
            print(f"skip {local} (not found)", flush=True)
            continue
        mb = os.path.getsize(path) / 1e6
        remote, upload_path = local, path
        if mb > MAX_MB:
            upload_path = gzip_copy(path)
            remote = local + ".gz"
            rmb = os.path.getsize(upload_path) / 1e6
            if rmb > MAX_MB:
                print(f"skip {local} ({mb:.0f} MB, {rmb:.0f} MB gz) - too big for the API",
                      flush=True)
                continue
            message = f"{message} [uploaded gzipped: the API refuses files above ~42 MB]"
            print(f"uploading {local} ({mb:.1f} MB) as {remote} ({rmb:.1f} MB) ...",
                  flush=True)
        else:
            print(f"uploading {local} ({mb:.1f} MB) ...", flush=True)
        try:
            url = gh_api.upload(remote, upload_path, message)
            print("  ->", url, flush=True)
        except Exception as e:
            print(f"  !! failed: {e}", flush=True)


if __name__ == "__main__":
    publish(only=sys.argv[1:] or None)
