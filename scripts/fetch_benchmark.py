"""Download and extract Wang et al. 2017 SSVEP Benchmark subjects into data/benchmark/.

Source: Tsinghua BCI lab, http://bci.med.tsinghua.edu.cn/download.html
Each subject is a ~105 MB ``S{n}.mat.7z`` archive (~140 MB extracted).
Requires the ``data`` extra: ``pip install -e ".[data]"`` (py7zr).

Usage:
    python scripts/fetch_benchmark.py 1 2 3 [--data-dir data/benchmark]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://bci.med.tsinghua.edu.cn/upload/yijun/"
METADATA = ("Freq_Phase.mat", "64-channels.loc", "Readme.txt", "Sub_info.txt")


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"have {dst.name}")
        return
    print(f"downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:
        shutil.copyfileobj(r, fh)
    tmp.rename(dst)


def extract(archive: Path, dst_dir: Path) -> Path:
    import py7zr

    out = dst_dir / archive.name.removesuffix(".7z")
    if out.exists():
        print(f"have {out.name}")
        return out
    print(f"extracting {archive.name}")
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(dst_dir)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("subjects", type=int, nargs="+")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmark")
    args = ap.parse_args(argv)
    raw = args.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in METADATA:
        download(BASE_URL + name, args.data_dir / name)
    for s in args.subjects:
        if not 1 <= s <= 35:
            raise SystemExit(f"subject must be 1..35, got {s}")
        archive = raw / f"S{s}.mat.7z"
        download(BASE_URL + archive.name, archive)
        extract(archive, args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
