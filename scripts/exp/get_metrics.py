import argparse
import glob
import os

import pandas as pd
import recode


# ---------------------------------------------------------------------------
# Legacy: load from recode_result CSV files
# ---------------------------------------------------------------------------

def load_split_results(split, result_dir="./results"):
    pattern = os.path.join(result_dir, f"recode_result_{split}_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"[WARN] No files found for split={split}")
        return None

    print(f"\nProcessing split={split}")
    print(f"Found {len(csv_files)} files")
    for f in csv_files:
        print(f" - {f}")

    df_list = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df[["transformed_text", "type", "recode_result"]]

    return merged_df


def run_split(split, result_dir):
    """Legacy evaluation from recode_result CSV files."""
    final_df = load_split_results(split, result_dir=result_dir)

    if final_df is None:
        return

    final_df_filtered = final_df.dropna()
    print(f"final_df_filtered len : {len(final_df_filtered)}")

    results = recode.evaluate_re(final_df_filtered["type"], final_df_filtered["recode_result"])
    recode.print_results(results, name=f"{split}")


# ---------------------------------------------------------------------------
# New: evaluate merged TSV files against gold JSONL
# ---------------------------------------------------------------------------

def run_tsv_eval(gold_path, tsv_dir):
    """Evaluate TSV prediction files against gold JSONL."""
    test_dataset = recode.read_file(gold_path)
    gold = test_dataset["type"]

    tsv_files = sorted(glob.glob(os.path.join(tsv_dir, "*.tsv")))
    if not tsv_files:
        print(f"[WARN] No TSV files found in {tsv_dir}")
        return

    for tsv_path in tsv_files:
        name = os.path.splitext(os.path.basename(tsv_path))[0]
        merged_df = pd.read_csv(tsv_path, sep="\t")
        pred = merged_df["type"]

        results = recode.evaluate_re(gold, pred)
        recode.print_results(results, name=name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RECoDe evaluation metrics")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "all"],
        help="Evaluate recode_result CSV files for this split.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results",
        help="Directory containing recode_result CSV files.",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to gold JSONL file (e.g., test.jsonl).",
    )
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default=None,
        help="Directory containing prediction TSV files.",
    )
    args = parser.parse_args()

    if args.gold and args.tsv_dir:
        run_tsv_eval(args.gold, args.tsv_dir)
    elif args.split:
        if args.split == "all":
            for split in ["train", "val"]:
                run_split(split, args.result_dir)
        else:
            run_split(args.split, args.result_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
