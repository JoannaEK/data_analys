"""Poll the repo for tables that are not there yet: every 2 min, up to 45 min."""
import os
import sys
import time

import gh_api

TARGETS = [
    "features_part_1.csv",
    "1exp_opt_1.csv",
    "filt_feature.csv",
    "3exp_opt_1.csv",
]
INTERVAL = 120
TIMEOUT = 45 * 60

found = {}
deadline = time.time() + TIMEOUT
while time.time() < deadline and len(found) < len(TARGETS):
    for t in TARGETS:
        if t in found:
            continue
        try:
            meta = gh_api.get_meta(t)
        except Exception as e:
            print(f"error checking {t}: {e}", flush=True)
            continue
        if meta:
            gh_api.download(t, t)
            found[t] = meta
            print(f"FOUND {t} size={meta[1]} sha={meta[0]}", flush=True)
    missing = [t for t in TARGETS if t not in found]
    print(f"[{time.strftime('%H:%M:%S')}] missing: {missing}", flush=True)
    if missing and time.time() < deadline:
        time.sleep(INTERVAL)

print("DONE found:", sorted(found), flush=True)
print("STILL MISSING:", [t for t in TARGETS if t not in found], flush=True)
