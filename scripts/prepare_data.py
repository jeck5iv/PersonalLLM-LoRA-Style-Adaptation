import argparse
from pathlib import Path

import pandas as pd

from utils.text import clean_text


def split_posts(data, min_chars=100):
    if not {"date", "text"}.issubset(data.columns):
        raise ValueError("Input must contain date and text columns")
    work = pd.DataFrame({
        "date": pd.to_datetime(data["date"], utc=True, errors="raise"),
        "cleaned_text": data["text"].map(clean_text),
    })
    if work["date"].isna().any():
        raise ValueError("Every post needs a timestamp")
    work = work[work["cleaned_text"].str.len() >= min_chars]
    work = work.sort_values("date", kind="stable").drop_duplicates("cleaned_text")
    dates = work["date"].drop_duplicates().tolist()
    if len(dates) < 10:
        raise ValueError("At least ten distinct timestamps are needed after cleaning")
    train_end, val_end = dates[int(len(dates) * 0.8)], dates[int(len(dates) * 0.9)]
    return {
        "train": work[work["date"] < train_end].reset_index(drop=True),
        "val": work[(work["date"] >= train_end) & (work["date"] < val_end)].reset_index(drop=True),
        "test": work[work["date"] >= val_end].reset_index(drop=True),
    }


def prepare_data(paths, min_chars=100):
    splits = split_posts(pd.read_csv(paths.tg_messages), min_chars=min_chars)
    for name, frame in splits.items():
        path = getattr(paths, name + "_data")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"{name}: {len(frame)} posts")
    return splits


def main():
    parser = argparse.ArgumentParser(description="Prepare chronological text splits")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=100)
    args = parser.parse_args()
    splits = split_posts(pd.read_csv(args.input), min_chars=args.min_chars)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(args.output / f"{name}_data.csv", index=False)
        print(f"{name}: {len(frame)} posts")


if __name__ == "__main__":
    main()
