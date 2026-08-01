#!/usr/bin/env python3
"""Download all 8 BEIR datasets to data/beir/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS, BEIR_DATA_DIR, BEIR_URL
from beir import util


def main():
    BEIR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        dest = BEIR_DATA_DIR / dataset
        if dest.exists() and any(dest.iterdir()):
            print(f"[skip]  {dataset}  (already at {dest})")
            continue
        print(f"[download] {dataset} ...")
        url = BEIR_URL.format(dataset)
        util.download_and_unzip(url, str(BEIR_DATA_DIR))
        print(f"[done]  {dataset}")

    print(f"\nAll datasets in {BEIR_DATA_DIR}")


if __name__ == "__main__":
    main()
