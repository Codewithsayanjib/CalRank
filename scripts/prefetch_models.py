#!/usr/bin/env python3
"""Pre-download all 6 cross-encoder models to the HuggingFace cache.
Run this once before the overnight sessions so there are zero network
fetches during scoring.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]     = "1"

from config import MODELS, TRUST_REMOTE_CODE
from sentence_transformers import CrossEncoder

def main():
    print(f"Pre-downloading {len(MODELS)} models …\n")
    for model_key, model_name in MODELS.items():
        print(f"  [{model_key}]  {model_name}")
        kwargs = {}
        if model_key in TRUST_REMOTE_CODE:
            kwargs["trust_remote_code"] = True
        try:
            m = CrossEncoder(model_name, **kwargs)
            del m
            print(f"  [OK] {model_key}")
        except Exception as e:
            print(f"  [FAIL] {model_key}: {e}")
            sys.exit(1)
        print()
    print("All models cached. Ready for overnight run.")

if __name__ == "__main__":
    main()
