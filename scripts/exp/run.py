import argparse

import recode


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based relation extraction for CoDiet dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/annotation",
        help="Inference dataset path, should have jsonl files with split names",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (e.g., train, val, test)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8010/v1",
        help="Address for OpenAI compatible server",
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Name of the model to use"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the inference server",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Starting index for processing"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="Ending index for processing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    dataset = recode.get_transformed_datasets(args.data_path)

    if args.end_idx is not None:
        current_dataset = dataset[args.split].iloc[args.start_idx : args.end_idx + 1].copy()
    else:
        current_dataset = dataset[args.split].iloc[args.start_idx :].copy()

    output_list = []
    for idx, row in current_dataset.iterrows():
        result = recode.predict(
            row["Node1_str"],
            row["Node2_str"],
            row["transformed_text"],
            base_url=args.base_url,
            model_name=args.model_name,
            api_key=args.api_key,
            temperature=0.2,
            top_p=0.8,
            num_max_tokens=512,
            num_trials=3,
        )
        print(f"Processing idx {idx}...")
        print("sentence: ", row["transformed_text"])
        print("Predicted: ", result)
        print("Answer: ", row["type"])
        print("====================")

        output_list.append(result)

    current_dataset["recode_result"] = output_list
    recode.save_results(current_dataset, args.output_dir, args.split, args.model_name)


if __name__ == "__main__":
    main()
