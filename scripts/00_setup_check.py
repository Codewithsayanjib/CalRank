#!/usr/bin/env python3
"""Verify that the CalRank environment is ready before running experiments."""
import sys
import subprocess

ROOT = __import__("pathlib").Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [OK]   {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def main():
    print("=" * 55)
    print("  CalRank — Environment Check")
    print("=" * 55)
    failures = []

    # Python
    ok = check("Python", lambda: sys.version.split()[0])
    if not ok:
        failures.append("python")

    # PyTorch + MPS
    def _torch():
        import torch
        mps = torch.backends.mps.is_available()
        return f"torch {torch.__version__}  MPS={'YES' if mps else 'NO (CPU fallback)'}"
    ok = check("PyTorch + MPS", _torch)
    if not ok:
        failures.append("torch")

    # Required Python packages
    pkg_map = {
        "beir":                  "beir",
        "sentence_transformers": "sentence-transformers",
        "bm25s":                 "bm25s",
        "pandas":                "pandas",
        "pyarrow":               "pyarrow",
        "numpy":                 "numpy",
        "scipy":                 "scipy",
        "sklearn":               "scikit-learn",
        "matplotlib":            "matplotlib",
        "tqdm":                  "tqdm",
    }
    for import_name, pkg_name in pkg_map.items():
        def _pkg(n=import_name):
            m = __import__(n)
            return getattr(m, "__version__", "installed")
        ok = check(f"pkg: {pkg_name}", _pkg)
        if not ok:
            failures.append(pkg_name)

    # Data directories
    def _dirs():
        from config import BEIR_DATA_DIR, INDEX_DIR, RETRIEVAL_DIR, SCORES_DIR
        for d in [BEIR_DATA_DIR, INDEX_DIR, RETRIEVAL_DIR, SCORES_DIR]:
            assert d.exists(), f"missing {d}"
        return "ok"
    ok = check("Data directories", _dirs)
    if not ok:
        failures.append("dirs")

    print()
    if failures:
        print(f"[!] {len(failures)} check(s) failed: {failures}")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("[+] All checks passed. Ready to run CalRank pipeline.")
        print()
        print("    Step 1:  python scripts/01_download_beir.py")
        print("    Step 2:  python scripts/02_build_bm25_index.py")
        print("    Step 3:  python scripts/03_bm25_retrieve.py")
        print("    Session 1: bash run_session1.sh   (~3.5 hrs, small models)")
        print("    Session 2: bash run_session2.sh   (~4 hrs, BGE + analysis)")


if __name__ == "__main__":
    main()
