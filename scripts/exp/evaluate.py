"""evaluate.py - Gold JSONL + Prediction TSV → 점수 확인

Usage:
    python evaluate.py --gold test.jsonl --pred test_model_result.tsv
    python evaluate.py --gold test.jsonl --pred_dir ./results/real/
"""
import argparse
import glob
import os

import pandas as pd
import recode


def evaluate_single(gold_path, pred_path):
    """Evaluate a single prediction TSV against gold JSONL."""
    gold_df = recode.read_file(gold_path)
    pred_df = pd.read_csv(pred_path, sep="\t")

    gold = gold_df["type"]
    pred = pred_df["type"].fillna("association")

    if len(gold) != len(pred):
        print(f"[WARN] Length mismatch: gold={len(gold)}, pred={len(pred)}")

    name = os.path.splitext(os.path.basename(pred_path))[0]
    results = recode.evaluate_re(gold, pred)
    recode.print_results(results, name=name)
    return results


def evaluate_dir(gold_path, pred_dir):
    """Evaluate all TSV files in a directory against gold JSONL."""
    tsv_files = sorted(glob.glob(os.path.join(pred_dir, "*.tsv")))
    if not tsv_files:
        print(f"[WARN] No TSV files found in {pred_dir}")
        return

    # Skip full.tsv files (reference only)
    tsv_files = [f for f in tsv_files if not f.endswith("_full.tsv")]

    print(f"Gold: {gold_path}")
    print(f"Found {len(tsv_files)} prediction files in {pred_dir}\n")

    for tsv_path in tsv_files:
        evaluate_single(gold_path, tsv_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RECoDe predictions")
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Path to gold JSONL file (e.g., test.jsonl)",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Path to a single prediction TSV file",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Directory containing prediction TSV files (evaluates all)",
    )
    args = parser.parse_args()

    if args.pred:
        evaluate_single(args.gold, args.pred)
    elif args.pred_dir:
        evaluate_dir(args.gold, args.pred_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
