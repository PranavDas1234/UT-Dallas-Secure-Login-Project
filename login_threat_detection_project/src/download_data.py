"""
Download and extract the RBA login dataset.

Dataset:
Login Data Set for Risk-Based Authentication
Zenodo record: https://zenodo.org/records/6782156
"""

from pathlib import Path
import argparse
import zipfile
import requests


DATA_URL = "https://zenodo.org/records/6782156/files/rba-dataset.zip?download=1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "rba-dataset.zip"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[OK] ZIP already exists: {output_path}")
        return

    print(f"[INFO] Downloading dataset to {output_path}")
    print("[INFO] This file is large, so it may take a while.")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        percent = downloaded / total * 100
                        print(f"\rDownloaded: {percent:6.2f}%", end="")

    print("\n[OK] Download complete.")


def extract_zip(zip_path: Path, data_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find {zip_path}")

    print(f"[INFO] Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        print("[INFO] Files inside ZIP:")
        for name in names:
            print(f"  - {name}")

        z.extractall(data_dir)

    print(f"[OK] Extracted to {data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and only extract existing data/rba-dataset.zip"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_file(DATA_URL, ZIP_PATH)

    extract_zip(ZIP_PATH, DATA_DIR)


if __name__ == "__main__":
    main()
