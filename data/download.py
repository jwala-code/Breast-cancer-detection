"""
data/download.py
----------------
Downloads, extracts, and cleans the two UCI datasets used in this project:
  - Dataset A: Breast Cancer Wisconsin (Diagnostic) → data/wdbc.csv
  - Dataset B: Banknote Authentication              → data/banknote.csv

Run this script once before launching the notebook:
    python data/download.py

It is idempotent: re-running will not re-download if CSVs already exist.
"""

import os
import io
import zipfile
import requests
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "wdbc": {
        "url": "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        "zip_internal": "wdbc.data",
        "out_csv": os.path.join(DATA_DIR, "wdbc.csv"),
        "fallback_url": None,  # set below if needed
    },
    "banknote": {
        "url": "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        "zip_internal": "data_banknote_authentication.txt",
        "out_csv": os.path.join(DATA_DIR, "banknote.csv"),
        "fallback_url": None,
    },
}

# ── Column names ───────────────────────────────────────────────────────────────

# WDBC: id (dropped), diagnosis, then 30 numeric features
_WDBC_FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
WDBC_COLUMNS = ["id", "diagnosis"] + _WDBC_FEATURE_NAMES

# Banknote: 4 wavelet features + 1 target
BANKNOTE_COLUMNS = ["variance", "skewness", "curtosis", "entropy", "target"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _download_zip(url: str) -> bytes:
    """Download a ZIP file from a URL and return raw bytes."""
    print(f"  Downloading: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def _extract_file_from_zip(zip_bytes: bytes, filename: str) -> bytes:
    """Extract a single named file from a ZIP archive given as bytes."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Search case-insensitively in case the archive layout varies
        names = zf.namelist()
        matched = [n for n in names if os.path.basename(n) == filename]
        if not matched:
            raise FileNotFoundError(
                f"'{filename}' not found in archive. Contents: {names}"
            )
        return zf.read(matched[0])


def _try_ucimlrepo_wdbc() -> pd.DataFrame:
    """Fallback: load WDBC via the ucimlrepo package."""
    from ucimlrepo import fetch_ucirepo  # type: ignore
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    df = df.rename(columns={"Diagnosis": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def _try_ucimlrepo_banknote() -> pd.DataFrame:
    """Fallback: load Banknote Authentication via the ucimlrepo package."""
    from ucimlrepo import fetch_ucirepo  # type: ignore
    dataset = fetch_ucirepo(id=267)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = BANKNOTE_COLUMNS
    return df


# ── Main download functions ────────────────────────────────────────────────────

def download_wdbc(out_path: str) -> None:
    """Download and clean the Breast Cancer Wisconsin dataset."""
    cfg = DATASETS["wdbc"]
    try:
        zip_bytes = _download_zip(cfg["url"])
        raw = _extract_file_from_zip(zip_bytes, cfg["zip_internal"])
        df = pd.read_csv(
            io.StringIO(raw.decode("utf-8")),
            header=None,
            names=WDBC_COLUMNS,
        )
    except Exception as e:
        print(f"  Primary URL failed ({e}). Trying ucimlrepo fallback...")
        df = _try_ucimlrepo_wdbc()
        # Ensure standard column structure
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        df.to_csv(out_path, index=False)
        print(f"  [fallback] Saved {len(df)} rows → {out_path}")
        return

    # Drop ID column (column 0), encode diagnosis
    df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")


def download_banknote(out_path: str) -> None:
    """Download and clean the Banknote Authentication dataset."""
    cfg = DATASETS["banknote"]
    try:
        zip_bytes = _download_zip(cfg["url"])
        raw = _extract_file_from_zip(zip_bytes, cfg["zip_internal"])
        df = pd.read_csv(
            io.StringIO(raw.decode("utf-8")),
            header=None,
            names=BANKNOTE_COLUMNS,
        )
    except Exception as e:
        print(f"  Primary URL failed ({e}). Trying ucimlrepo fallback...")
        df = _try_ucimlrepo_banknote()

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=== Dataset Downloader ===")

    # --- WDBC ---
    wdbc_path = DATASETS["wdbc"]["out_csv"]
    if os.path.exists(wdbc_path):
        df = pd.read_csv(wdbc_path)
        print(f"[WDBC] Already exists — {len(df)} rows loaded from {wdbc_path}")
    else:
        print("[WDBC] Downloading...")
        download_wdbc(wdbc_path)
        df = pd.read_csv(wdbc_path)
        print(f"[WDBC] ✓ {len(df)} rows loaded.")

    # --- Banknote ---
    bn_path = DATASETS["banknote"]["out_csv"]
    if os.path.exists(bn_path):
        df = pd.read_csv(bn_path)
        print(f"[Banknote] Already exists — {len(df)} rows loaded from {bn_path}")
    else:
        print("[Banknote] Downloading...")
        download_banknote(bn_path)
        df = pd.read_csv(bn_path)
        print(f"[Banknote] ✓ {len(df)} rows loaded.")

    print("=== Done ===")


if __name__ == "__main__":
    main()
