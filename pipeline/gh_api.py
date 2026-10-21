"""GitHub API helper: list / download / upload files in the data_analys repo."""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO = "JoannaEK/data_analys"
API = f"https://api.github.com/repos/{REPO}"
TOKEN = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gh_token.txt"),
             encoding="utf-8").read().strip()


def _request(url, method="GET", payload=None, accept="application/vnd.github+json", timeout=120):
    data = None
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "opencode-agent",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, body


def list_files():
    status, body = _request(f"{API}/contents/")
    items = json.loads(body)
    return [(i["name"], i["type"], i.get("size", 0), i["sha"]) for i in items]


def get_meta(path):
    """Return (sha, size) for a path, or None if it does not exist."""
    url = f"{API}/contents/{path}"
    try:
        status, body = _request(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    j = json.loads(body)
    return j["sha"], j.get("size", 0)


def download(path, dest=None, retries=4):
    """Download raw file content. Returns True on success."""
    for attempt in range(retries):
        try:
            status, body = _request(f"{API}/contents/{path}",
                                    accept="application/vnd.github.raw+json")
            if dest:
                with open(dest, "wb") as f:
                    f.write(body)
            else:
                sys.stdout.write(body.decode("utf-8", errors="replace"))
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    return False


def upload(path, local, message, branch="main"):
    raw = open(local, "rb").read()
    b64 = base64.b64encode(raw).decode("ascii")
    payload = {"message": message, "content": b64, "branch": branch}
    meta = get_meta(path)
    if meta:
        payload["sha"] = meta[0]
    status, body = _request(f"{API}/contents/{path}", method="PUT", payload=payload)
    j = json.loads(body)
    return j.get("content", {}).get("html_url")


def wait_for(path, timeout=2700, interval=120):
    """Poll every `interval` seconds until the path appears. Returns sha or None."""
    deadline = time.time() + timeout
    waited = 0
    while True:
        meta = get_meta(path)
        if meta:
            print(f"FOUND {path} sha={meta[0]} after {waited}s")
            return meta[0]
        if time.time() >= deadline:
            print(f"TIMEOUT {path} after {waited}s")
            return None
        print(f"not found yet: {path} (waited {waited}s)", flush=True)
        time.sleep(interval)
        waited += interval


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for name, typ, size, sha in sorted(list_files()):
            print(f"{size:>12}  {typ:<5}  {name}")
    elif cmd == "get":
        path = sys.argv[2]
        dest = sys.argv[3] if len(sys.argv) > 3 else None
        ok = download(path, dest)
        print("OK" if ok else "NOT_FOUND")
    elif cmd == "put":
        print(upload(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "wait":
        wait_for(sys.argv[2],
                 timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 2700,
                 interval=int(sys.argv[4]) if len(sys.argv) > 4 else 120)
