"""Step 1: edit the retail table per the task spec.

1) keep only rows whose StockCode starts with a digit
2) StockCode -> itemid
3) Customer ID -> visitor
4) drop Invoice, Description
5) Country -> visitor_cat, encoded as a stable numeric id
6) price -> item_cat (item_cat_1..item_cat_10) by quantile bins,
   after removing min/max outliers (rule: extreme differs >2x from all other
   values AND occurs at most 3 times); price column is dropped
"""
import json
import re

import numpy as np
import pandas as pd

SRC = "online_retail_II.csv"
DST = "retail.csv"
META = "retail_meta.json"

df = pd.read_csv(
    SRC,
    dtype={"Invoice": "string", "StockCode": "string", "Customer ID": "string"},
    parse_dates=["InvoiceDate"],
)
print("loaded:", df.shape)

# --- 1) filter: StockCode starts with a digit -------------------------------
mask = df["StockCode"].str.match(r"^\d", na=False)
print("dropped (non-digit StockCode):", (~mask).sum())
df = df.loc[mask].copy()
df["StockCode"] = df["StockCode"].str.strip()

# --- 2) / 3) renames ---------------------------------------------------------
df = df.rename(columns={"StockCode": "itemid", "Customer ID": "visitor"})

# --- 4) drop Invoice, Description -------------------------------------------
df = df.drop(columns=["Invoice", "Description"])

# customer id normalisation: "13085.0" -> "13085"
v = pd.to_numeric(df["visitor"], errors="coerce")
df["visitor"] = v.astype("Int64").astype("string")
print("rows without visitor:", df["visitor"].isna().sum())

# --- 5) Country -> visitor_cat (stable numeric id) --------------------------
country_to_id = {c: i for i, c in enumerate(sorted(df["Country"].unique()))}
df["visitor_cat"] = df["Country"].map(country_to_id).astype("int32")
print("countries:", len(country_to_id))

df["Price"] = df["Price"].astype(float)

# --- 6.2) min/max outlier removal on price ----------------------------------
def extremes(series):
    return series.min(), series.max()


def second_below_max(s):
    m = s.max()
    below = s[s < m]
    return below.max() if len(below) else np.nan


def second_above_min(s):
    m = s.min()
    above = s[s > m]
    return above.min() if len(above) else np.nan


removed = []
for _ in range(5):
    mx = df["Price"].max()
    if df["Price"].eq(mx).sum() <= 3:
        smx = second_below_max(df["Price"])
        if pd.notna(smx) and mx > 2 * smx:
            n = int(df["Price"].eq(mx).sum())
            removed.append({"side": "max", "value": float(mx), "rows": n})
            df = df.loc[df["Price"] < mx].copy()
            continue
    mn = df["Price"].min()
    if df["Price"].eq(mn).sum() <= 3:
        smn = second_above_min(df["Price"])
        if pd.notna(smn) and smn > 2 * mn:
            n = int(df["Price"].eq(mn).sum())
            removed.append({"side": "min", "value": float(mn), "rows": n})
            df = df.loc[df["Price"] > mn].copy()
            continue
    break

print("price outliers removed:", removed)
pmin, pmax = extremes(df["Price"])
print(f"price after cleaning: min={pmin} max={pmax} rows={len(df)}")

# --- 6.3-6.4) quantile bins -> item_cat_1..item_cat_10 ----------------------
labels = [f"item_cat_{i}" for i in range(1, 11)]
df["item_cat"] = pd.qcut(df["Price"], q=10, labels=labels, duplicates="drop").astype("string")
df["item_cat"] = df["item_cat"].fillna(labels[0])

share = df["item_cat"].value_counts(normalize=True).sort_index()
print("item_cat share (must be 5%-20%):")
print((share * 100).round(2).to_string())
assert share.min() >= 0.05 and share.max() <= 0.20, "bin share outside 5%-20%"

bounds = {}
for i in labels:
    sel = df.loc[df["item_cat"] == i, "Price"]
    bounds[i] = {"min": float(sel.min()), "max": float(sel.max()),
                 "count": int(len(sel)), "share": float(share[i])}

# --- final layout ------------------------------------------------------------
out = df[["itemid", "visitor", "visitor_cat", "Quantity", "InvoiceDate", "item_cat"]].copy()
out["itemid"] = out["itemid"].astype("string")
out = out.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
out.to_csv(DST, index=False, encoding="utf-8", lineterminator="\r\n")

meta = {
    "source": SRC,
    "output": DST,
    "rows": int(len(out)),
    "removed_price_outliers": removed,
    "price_min": float(pmin),
    "price_max": float(pmax),
    "visitor_cat_map": country_to_id,
    "item_cat_bounds": bounds,
}
with open(META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print("saved:", DST, out.shape)
print(out.head(5).to_string())
print("items:", out["itemid"].nunique(), "visitors:", out["visitor"].nunique(),
      "visitor_cat:", out["visitor_cat"].nunique())
