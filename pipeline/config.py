"""Pipeline configuration and conventions.

Conventions (from the task):
  t            - snapshot time, produced by the pipeline, not a data column
  horizon      - length of the target window, 7 days
  W(t, w)      - half-open feature window [t - w, t)
  n, m, k, q, s, u, x, z - window lengths (hyper-parameters, never hardcoded)
  denominator 0 -> feature value 0
  all aggregations vectorised (no python row loops)
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RETAIL = os.path.join(HERE, "retail.csv")
CONFIG_PATH = os.path.join(HERE, "config.json")

HORIZON = 7
PARAM_MIN = 7

# window-length hyper-parameters, per feature group
WINDOW_PARAMS = {
    "user": "n",       # group 1
    "item": "m",       # group 2
    "item_cat": "k",   # group 3.1
    "visitor_cat": "x",  # group 3.2
    "vi": "q",         # group 4
    "vc": "s",         # group 5.1
    "ic": "z",         # group 5.2
    "ii": "u",         # group 6
}


def build_config(retail=RETAIL):
    df = pd.read_csv(retail, usecols=["InvoiceDate"], parse_dates=["InvoiceDate"])
    obs_start = df["InvoiceDate"].min().normalize()
    obs_end = df["InvoiceDate"].max().normalize()
    y = int((obs_end - obs_start).days)          # Y - whole observation time, in days
    cfg = {
        "obs_start": obs_start.strftime("%Y-%m-%d"),
        "obs_end": obs_end.strftime("%Y-%m-%d"),
        "n_days": y + 1,
        "Y": y,
        "param_min": PARAM_MIN,
        "param_max": y,
        "param_mean": int(round((PARAM_MIN + y) / 2)),
        "horizon": HORIZON,
        "window_params": WINDOW_PARAMS,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    return cfg


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return build_config()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def snapshot_times(cfg, stride=1):
    """Day boundaries t = midnight at the end of each observed day.

    W(t, w) = [t - w, t) then covers exactly the whole days [d+1-w, d+1).
    """
    start = pd.Timestamp(cfg["obs_start"])
    n = cfg["n_days"]
    days = list(range(0, n, stride))
    if stride > 1 and days[-1] != n - 1:
        days.append(n - 1)
    return [start + pd.Timedelta(days=d + 1) for d in days], days


if __name__ == "__main__":
    cfg = build_config()
    print(json.dumps(cfg, ensure_ascii=False, indent=1))
    ts, days = snapshot_times(cfg)
    print("snapshots:", len(ts), ts[0], "->", ts[-1])
