import os
import shutil
import gdown
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DL_DIR = DATA_DIR / "_dl"

FOLDER_IDS = [
    "12UDbV2YKf7ox5Ccxuxpk49MtxlrKPDcp", # medi
    "1cQ5k3jATUBV6Sz34C-GOnKGOOiZQFwPe", # plank
    "1LYBvZgsehhhe6M0W0jyD2-rxRrBTRhYE", # math
]

def main():
    print("Preparing directories...")
    # Move existing txt files to data/_old
    old_dir = DATA_DIR / "_old"
    old_dir.mkdir(parents=True, exist_ok=True)
    for txt_file in DATA_DIR.glob("*.txt"):
        try:
            shutil.move(str(txt_file), str(old_dir / txt_file.name))
        except Exception as e:
            print(f"Error moving {txt_file.name}: {e}")

    DL_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading folders from Google Drive...")
    for folder_id in FOLDER_IDS:
        print(f"Downloading folder: {folder_id}")
        try:
            gdown.download_folder(id=folder_id, output=str(DL_DIR), quiet=False, use_cookies=False)
        except Exception as e:
            print(f"Error downloading {folder_id}: {e}")

    print("Flattening files...")
    # Move all *.txt files from _dl subdirectories to data/
    count = 0
    for txt_path in DL_DIR.rglob("*.txt"):
        dest = DATA_DIR / txt_path.name
        try:
            shutil.move(str(txt_path), str(dest))
            count += 1
            print(f"Moved: {txt_path.name}")
        except Exception as e:
            print(f"Error moving {txt_path.name}: {e}")

    # Remove the temporary download folder
    if DL_DIR.exists():
        shutil.rmtree(DL_DIR)

    print(f"Download complete! Flattened {count} files into {DATA_DIR}.")

if __name__ == "__main__":
    main()

