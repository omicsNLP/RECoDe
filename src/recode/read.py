import json
import pandas as pd


def read_file(file_path):
    parsed_list = []
    for line in open(file_path, "r"):
        parsed_list.append(json.loads(line))

    df = pd.DataFrame(parsed_list)
    return df


def get_datasets(dir):
    return {
        "train": read_file(f"{dir}/train.jsonl"),
        "val": read_file(f"{dir}/val.jsonl"),
        # 'test': read_file(f'{dir}/test.jsonl'),
    }


def _get_transformed_text_(text, node1_start, node1_length, node2_start, node2_length):
    # Extract start positions and lengths
    e1_str = text[node1_start : node1_start + node1_length]
    e2_str = text[node2_start : node2_start + node2_length]

    # Determine order of insertion based on offsets
    if node1_start < node2_start:
        # Insert Node2 first (later position), then Node1
        text = (
            text[:node2_start]
            + f"<entity2>{e2_str}</entity2>"
            + text[node2_start + node2_length :]
        )
        text = (
            text[:node1_start]
            + f"<entity1>{e1_str}</entity1>"
            + text[node1_start + node1_length :]
        )
    else:
        # Insert Node1 first, then Node2
        text = (
            text[:node1_start]
            + f"<entity1>{e1_str}</entity1>"
            + text[node1_start + node1_length :]
        )
        text = (
            text[:node2_start]
            + f"<entity2>{e2_str}</entity2>"
            + text[node2_start + node2_length :]
        )

    return text


def _get_transformed_text_entity_one_(text, node1_start, node1_length):
    # Extract start positions and lengths
    e1_str = text[node1_start : node1_start + node1_length]

    # Determine order of insertion based on offsets
    text = (
        text[:node1_start]
        + f"<entity>{e1_str}</entity>"
        + text[node1_start + node1_length :]
    )
    return text


def get_transformed_text(row):
    return _get_transformed_text_(
        row["original_text"],
        row["Node1_offset"],
        row["Node1_length"],
        row["Node2_offset"],
        row["Node2_length"],
    )


def get_transformed_text_entity1(row):
    return _get_transformed_text_entity_one_(
        row["original_text"], row["Node1_offset"], row["Node1_length"]
    )


def get_transformed_text_entity2(row):
    return _get_transformed_text_entity_one_(
        row["original_text"], row["Node2_offset"], row["Node2_length"]
    )


def get_transformed_datasets(datasets):
    if type(datasets) == str:
        datasets = get_datasets(datasets)

    datasets["train"]["transformed_text"] = datasets["train"].apply(
        get_transformed_text, axis=1
    )
    datasets["val"]["transformed_text"] = datasets["val"].apply(
        get_transformed_text, axis=1
    )
    # datasets['test']['transformed_text'] = datasets['test'].apply(get_transformed_text, axis=1)

    datasets["train"]["transformed_text_entity1"] = datasets["train"].apply(
        get_transformed_text_entity1, axis=1
    )
    datasets["val"]["transformed_text_entity1"] = datasets["val"].apply(
        get_transformed_text_entity1, axis=1
    )
    # datasets['test']['transformed_text_entity1'] = datasets['test'].apply(get_transformed_text_entity1, axis=1)

    datasets["train"]["transformed_text_entity2"] = datasets["train"].apply(
        get_transformed_text_entity2, axis=1
    )
    datasets["val"]["transformed_text_entity2"] = datasets["val"].apply(
        get_transformed_text_entity2, axis=1
    )
    # datasets['test']['transformed_text_entity2'] = datasets['test'].apply(get_transformed_text_entity2, axis=1)

    return datasets
