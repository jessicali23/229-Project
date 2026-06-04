"""
data/download_datasets.py

Downloads RAVDESS, TESS, and EmoDB into a local directory.

Usage:
    python data/download_datasets.py --datasets ravdess tess emodb --output_dir ./data/raw
"""

import argparse
import os
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import List
from urllib.request import urlretrieve
from tqdm import tqdm


# ─── Metadata ─────────────────────────────────────────────────────────────────

DATASETS = {
    "ravdess": {
        "description": "Ryerson Audio-Visual Database of Emotional Speech and Song",
        "url": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        "filename": "ravdess.zip",
        "extract": True,
    },
    "tess": {
        "description": "Toronto Emotional Speech Set",
        # TESS is hosted on UofT Dataverse; direct download requires browser auth.
        # We provide a mirror fallback or manual instruction.
        "url": None,
        "filename": "tess.zip",
        "manual": (
            "Download TESS manually from:\n"
            "  https://tspace.library.utoronto.ca/handle/1807/24487\n"
            "Place the zip at: <output_dir>/tess.zip"
        ),
        "extract": True,
    },
    "emodb": {
        "description": "Berlin Database of Emotional Speech",
        "url": "http://emodb.bilderbar.info/download/download.zip",
        "filename": "emodb.zip",
        "extract": True,
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

class _ProgressHook:
    def __init__(self, desc: str):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=self.desc,
            )
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def _download(url: str, dest: Path, desc: str = "Downloading") -> None:
    hook = _ProgressHook(desc)
    urlretrieve(url, str(dest), reporthook=hook)


def _extract(archive: Path, dest_dir: Path) -> None:
    print(f"  Extracting {archive.name} …")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest_dir)
    elif archive.suffix in (".tar", ".gz", ".bz2", ".tgz"):
        with tarfile.open(archive) as t:
            t.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive.suffix}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def download_datasets(datasets: List[str], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        if name not in DATASETS:
            print(f"[WARN] Unknown dataset '{name}'. Skipping.")
            continue

        info = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"  {name.upper()} — {info['description']}")
        print(f"{'='*60}")

        archive = out / info["filename"]
        dataset_dir = out / name

        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"  Already extracted at {dataset_dir}. Skipping.")
            continue

        if info.get("manual"):
            print(f"  [MANUAL DOWNLOAD REQUIRED]\n  {info['manual']}")
            if not archive.exists():
                continue

        if info["url"] and not archive.exists():
            print(f"  Downloading from {info['url']}")
            try:
                _download(info["url"], archive, desc=f"  {name}")
            except Exception as e:
                print(f"  [ERROR] Download failed: {e}")
                print("  Please download manually and place the archive at:")
                print(f"    {archive}")
                continue

        if archive.exists() and info.get("extract"):
            dataset_dir.mkdir(exist_ok=True)
            _extract(archive, dataset_dir)
            print(f"  Extracted to {dataset_dir}")
        else:
            print(f"  Archive not found at {archive}")

    print("\nDone. Dataset layout:")
    for p in sorted(out.iterdir()):
        print(f"  {p}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download speech emotion datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ravdess", "emodb"],
        choices=list(DATASETS.keys()),
        help="Datasets to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/raw",
        help="Directory to save downloaded data",
    )
    args = parser.parse_args()
    download_datasets(args.datasets, args.output_dir)
