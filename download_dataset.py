"""
Download the COVID-19 Fake News Dataset from Kaggle.

Usage:
    python download_dataset.py

Requirements:
    - pip install kaggle
    - Place your kaggle.json API token in ~/.kaggle/ (Linux/macOS)
      or C:\\Users\\<username>\\.kaggle\\ (Windows).
      Download from: https://www.kaggle.com/settings -> API -> Create New Token
"""

import os
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_SLUG = "elroyggj/covid19-fake-news-dataset-nlp"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

MANUAL_INSTRUCTIONS = """
══════════════════════════════════════════════════════════════════════
  Kaggle API is not configured or not installed.
══════════════════════════════════════════════════════════════════════

  To download the dataset automatically, follow these steps:

  1. Install the Kaggle package:
         pip install kaggle

  2. Go to https://www.kaggle.com/settings and click
     "API → Create New Token" to download kaggle.json.

  3. Place kaggle.json in the correct location:
       • Linux / macOS : ~/.kaggle/kaggle.json
       • Windows       : C:\\Users\\<username>\\.kaggle\\kaggle.json

  4. Re-run this script:
         python download_dataset.py

  ── Manual alternative ──────────────────────────────────────────────
  Visit the dataset page directly and download the ZIP:
      https://www.kaggle.com/datasets/elroyggj/covid19-fake-news-dataset-nlp

  Extract the contents into:
      {data_dir}
══════════════════════════════════════════════════════════════════════
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _ensure_directory(path: Path) -> None:
    """Create the directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {path}")


def _extract_zip_files(directory: Path) -> None:
    """Extract any .zip files found inside *directory*."""
    for zip_path in directory.glob("*.zip"):
        print(f"[INFO] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(directory)
        print(f"[INFO] Extracted to {directory}")


def download_dataset() -> None:
    """Download the dataset using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(MANUAL_INSTRUCTIONS.format(data_dir=DATA_DIR))
        sys.exit(1)

    # Authenticate
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        print(f"[ERROR] Kaggle authentication failed: {exc}\n")
        print(MANUAL_INSTRUCTIONS.format(data_dir=DATA_DIR))
        sys.exit(1)

    _ensure_directory(DATA_DIR)

    print(f"[INFO] Downloading dataset '{DATASET_SLUG}' ...")
    try:
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(DATA_DIR),
            unzip=True,
        )
        print("[INFO] Download complete.")
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        print(f"\n[INFO] Trying alternative: downloading as ZIP ...")
        try:
            api.dataset_download_files(
                DATASET_SLUG,
                path=str(DATA_DIR),
                unzip=False,
            )
            _extract_zip_files(DATA_DIR)
            print("[INFO] Download and extraction complete.")
        except Exception as exc2:
            print(f"[ERROR] Alternative download also failed: {exc2}")
            print(MANUAL_INSTRUCTIONS.format(data_dir=DATA_DIR))
            sys.exit(1)

    # List downloaded files
    files = list(DATA_DIR.iterdir())
    if files:
        print(f"\n[INFO] Files in {DATA_DIR}:")
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"       {f.name:30s}  ({size_kb:,.1f} KB)")
    else:
        print("[WARNING] No files found after download.")

    print("\n[DONE] Dataset is ready at:", DATA_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    download_dataset()
