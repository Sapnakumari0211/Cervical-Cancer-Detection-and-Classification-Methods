"""Prepare SIPaKMeD dataset for two tasks:
1. Detection: Normal vs Abnormal
2. Classification: five cervical cell classes

Expected raw structure:
raw_dataset/
  im_Dyskeratotic/*.bmp
  im_Koilocytotic/*.bmp
  im_Metaplastic/*.bmp
  im_Parabasal/*.bmp
  im_Superficial-Intermediate/*.bmp

Output structure:
processed_dataset/
  detection/train|val|test/Normal|Abnormal
  classification/train|val|test/<five classes>
"""
import argparse
import random
import shutil
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
from config import BINARY_MAP, SEED

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def normalize_class_name(name: str) -> str:
    return name.replace("im_", "")


def copy_files(files: List[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, target_dir / file.name)


def main(raw_dir: str, output_dir: str) -> None:
    random.seed(SEED)
    raw = Path(raw_dir)
    out = Path(output_dir)
    if not raw.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {raw}")

    for class_folder in sorted([p for p in raw.iterdir() if p.is_dir()]):
        original_name = class_folder.name
        clean_name = normalize_class_name(original_name)
        binary_label = BINARY_MAP.get(original_name) or BINARY_MAP.get(clean_name)
        if binary_label is None:
            print(f"Skipping unknown class folder: {original_name}")
            continue

        files = [p for p in class_folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if not files:
            print(f"No images found in {class_folder}")
            continue

        train_val, test = train_test_split(files, test_size=0.20, random_state=SEED)
        train, val = train_test_split(train_val, test_size=0.25, random_state=SEED)  # 60/20/20

        splits = {"train": train, "val": val, "test": test}
        for split, split_files in splits.items():
            copy_files(split_files, out / "classification" / split / clean_name)
            copy_files(split_files, out / "detection" / split / binary_label)

    print(f"Prepared dataset saved at: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Path to raw SIPaKMeD folder")
    parser.add_argument("--output_dir", required=True, help="Path to output processed folder")
    args = parser.parse_args()
    main(args.raw_dir, args.output_dir)
