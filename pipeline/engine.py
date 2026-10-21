"""Vectorised engine for windowed aggregations.

A snapshot t is a day boundary, so W(t, w) = [t-w, t) maps exactly onto the
day-index window [d+1-w, d+1).  All counts are answered with a single global
`np.searchsorted` over a (key, day) index - no python loops over rows.
"""
import numpy as np
import pandas as pd


def day_index(ts, origin):
    """Day index of each timestamp, 0 = the first observed day."""
    return ((ts - origin).dt.total_seconds() // 86400).astype(np.int64)


class DayCounter:
    """Transaction counts per (key, day) with O(log N) window queries."""

    def __init__(self, key_ids, days, n_keys, n_days):
        self.n_days = int(n_days)
        comb = key_ids.astype(np.int64) * self.n_days + days
        uniq, inv = np.unique(comb, return_inverse=True)
        counts = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
        prefix = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=prefix[1:])
        self.uniq = uniq
        self.prefix = prefix

        key_of_uniq = uniq // self.n_days
        first_occ = np.searchsorted(key_of_uniq, key_of_uniq, "left")
        base = np.zeros(n_keys, dtype=np.int64)
        present = np.zeros(n_keys, dtype=bool)
        base[key_of_uniq[first_occ]] = prefix[first_occ]
        present[key_of_uniq[first_occ]] = True
        self.base = base
        self.present = present

    def count_before(self, q_key, q_day):
        """Transactions of the key with day index < q_day."""
        q_day = np.maximum(q_day, 0)
        pos = np.searchsorted(self.uniq, q_key * self.n_days + q_day, "left")
        out = self.prefix[pos] - self.base[q_key]
        return np.where(self.present[q_key], out, 0)

    def window_count(self, q_key, q_day_end, window):
        """Transactions of the key inside [q_day_end - window, q_day_end)."""
        return (self.count_before(q_key, q_day_end)
                - self.count_before(q_key, q_day_end - window))

    def active_days(self, q_key, q_day_end, window):
        """Distinct active days of the key inside the window."""
        lo = np.maximum(q_day_end - window, 0)
        pos_lo = np.searchsorted(self.uniq, q_key * self.n_days + lo, "left")
        pos_hi = np.searchsorted(self.uniq, q_key * self.n_days + q_day_end, "left")
        return np.where(self.present[q_key], pos_hi - pos_lo, 0)

    def active_day_gap(self, q_key, q_day_end, window):
        """mean(diff(sorted(active days))) inside the window; 0 when |A| < 2.

        The gaps telescope, so the mean gap is (last - first) / (n_days - 1).
        """
        lo = np.maximum(q_day_end - window, 0)
        pos_lo = np.searchsorted(self.uniq, q_key * self.n_days + lo, "left")
        pos_hi = np.searchsorted(self.uniq, q_key * self.n_days + q_day_end, "left")
        n = pos_hi - pos_lo
        first = np.where(n > 0, self.uniq[np.minimum(pos_lo, len(self.uniq) - 1)] % self.n_days, 0)
        last = np.where(n > 1, self.uniq[np.maximum(pos_hi - 1, 0)] % self.n_days, 0)
        out = np.where(n >= 2, (last - first) / np.maximum(n - 1, 1), 0.0)
        return np.where(self.present[q_key], out, 0.0)


def all_pairs_within_groups(group_ids, items):
    """All unordered pairs of items taken inside each group.

    group_ids: group id per element (need not be sorted)
    items:     item id per element (same length)
    Returns (a, b): item values of every pair, with a < b.
    Fully vectorised (sort + repeat/cumsum trick), no python loops.
    """
    n = len(items)
    order = np.lexsort((items, group_ids))      # by group, then by item value
    g = group_ids[order]
    it = items[order]
    uniq, start = np.unique(g, return_index=True)
    sizes = np.diff(np.append(start, n)).astype(np.int64)
    local = np.arange(n, dtype=np.int64) - np.repeat(start, sizes)

    # an element is the smaller half of (m-1-local) pairs
    cnt = np.repeat(sizes, sizes) - 1 - local
    gstart = np.repeat(start, sizes)            # global index of each group head
    total = int(cnt.sum())
    first = np.repeat(np.arange(n, dtype=np.int64), cnt)
    offs = np.repeat(np.cumsum(cnt) - cnt, cnt)
    second = np.repeat(gstart + local + 1, cnt) + (np.arange(total, dtype=np.int64) - offs)
    return it[first], it[second]


def encode_pairs(a, b, n_items):
    """Encode an unordered item pair into a single int64."""
    return a.astype(np.int64) * (int(n_items) + 1) + b.astype(np.int64)


def decode_pairs(pair, n_items):
    return (pair // (n_items + 1)).astype(np.int64), (pair % (n_items + 1)).astype(np.int64)


def co_purchase_pairs(visitor, item, n_items, min_users=2):
    """Item pairs bought by at least `min_users` distinct visitors.

    visitor/item: aligned int arrays of unique (visitor, item) pairs.
    Returns (item_a, item_b, n_users) with item_a < item_b.
    """
    order = np.lexsort((item, visitor))
    v = visitor[order]
    it = item[order]
    a, b = all_pairs_within_groups(v, it)
    if len(a) == 0:
        return (np.array([], dtype=np.int64),) * 3
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    pair = encode_pairs(lo, hi, n_items)
    uniq_pair, counts = np.unique(pair, return_counts=True)
    keep = counts >= min_users
    uniq_pair = uniq_pair[keep]
    counts = counts[keep]
    pa, pb = decode_pairs(uniq_pair, n_items)
    return pa, pb, counts


def mean_active_day_gap(day_lists):
    """mean(diff(sorted(active days))) per group; 0 when |A| < 2."""
    out = np.zeros(len(day_lists), dtype=np.float64)
    for i, days in enumerate(day_lists):
        if len(days) >= 2:
            out[i] = np.diff(days).mean()
    return out
