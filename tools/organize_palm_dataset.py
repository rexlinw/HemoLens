#!/usr/bin/env python3
"""Sort flat palm images into Anemic/ and Non-Anemic/ by filename."""

import shutil
from pathlib import Path

PALM_ROOT = Path(__file__).resolve().parent.parent / "data" / "palms"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def classify(filename: str) -> str | None:
    lower = filename.lower()
    if lower.startswith("non-anemic") or lower.startswith("non_anemic"):
        return "Non-Anemic"
    if lower.startswith("anemic"):
        return "Anemic"
    return None


def main():
    anemic_dir = PALM_ROOT / "Anemic"
    non_anemic_dir = PALM_ROOT / "Non-Anemic"
    anemic_dir.mkdir(parents=True, exist_ok=True)
    non_anemic_dir.mkdir(parents=True, exist_ok=True)

    moved = {"Anemic": 0, "Non-Anemic": 0, "skipped": 0, "already": 0}

    for path in PALM_ROOT.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue

        label = classify(path.name)
        if label is None:
            moved["skipped"] += 1
            print(f"  skip (unknown): {path.name}")
            continue

        dest_dir = anemic_dir if label == "Anemic" else non_anemic_dir
        dest = dest_dir / path.name
        if dest.exists():
            moved["already"] += 1
            path.unlink()
            continue
        shutil.move(str(path), str(dest))
        moved[label] += 1

    print(f"\nAnemic:      {moved['Anemic']} moved -> {anemic_dir}")
    print(f"Non-Anemic:  {moved['Non-Anemic']} moved -> {non_anemic_dir}")
    print(f"Already in folder (removed dup from root): {moved['already']}")
    print(f"Skipped:     {moved['skipped']}")
    print(f"Anemic count:     {len(list(anemic_dir.glob('*')))}")
    print(f"Non-Anemic count: {len(list(non_anemic_dir.glob('*')))}")


if __name__ == "__main__":
    main()
