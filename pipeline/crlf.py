"""Normalise a CSV file to the convention of the tables in the repository:
CRLF line endings, UTF-8 without BOM.

The pipeline writes the files with pandas (LF by default); this utility
converts them in place without loading the whole file into memory.
"""
import gzip
import os
import shutil
import sys


def to_crlf(path, gz=False):
    tmp = path + ".tmp"
    opener_in = gzip.open if gz else open
    opener_out = gzip.open if gz else open
    with opener_in(path, "rb") as fin, opener_out(tmp, "wb") as fout:
        buf = b""
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            # keep the last byte in case a CRLF is split between two chunks
            parts = buf.split(b"\n")
            buf = parts.pop()
            out = []
            for p in parts:
                p = p.rstrip(b"\r")
                out.append(p + b"\r\n")
            fout.write(b"".join(out))
        if buf:
            buf = buf.rstrip(b"\r")
            fout.write(buf + b"\r\n")
    shutil.move(tmp, path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        gz = p.endswith(".gz")
        to_crlf(p, gz=gz)
        print("crlf:", p, os.path.getsize(p))
