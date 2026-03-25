# %%
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Run inference by file")
parser.add_argument(
    "--data_dir", type=str, default="../../data/extraction/input/", help=""
)
parser.add_argument(
    "--output_dir", type=str, default="../../data/extraction/output/", help=""
)
args = parser.parse_args()

import recode

target_tuple_types = [
    # food to disease
    ["dietMethod", "diseasePhenotype"],
    ["foodRelated", "diseasePhenotype"],
    # food to bio
    ["dietMethod", "geneSNP"],
    ["dietMethod", "proteinEnzyme"],
    ["dietMethod", "metabolites"],
    ["dietMethod", "microbiome"],
    ["foodRelated", "geneSNP"],
    ["foodRelated", "proteinEnzyme"],
    ["foodRelated", "metabolites"],
    ["foodRelated", "microbiome"],
    # disease to bio
    ["diseasePhenotype", "geneSNP"],
    ["diseasePhenotype", "proteinEnzyme"],
    ["diseasePhenotype", "metabolites"],
    ["diseasePhenotype", "microbiome"],
]

candidate_generator = recode.CoDietRelationCandidateGenerator(target_tuple_types)

input_path = args.data_dir


def make_key(e1, e2):
    # Build a tuple key based on offsets/lengths, sorted so (e1,e2) == (e2,e1)
    e1_key = (e1.locations[0].offset, e1.locations[0].length)
    e2_key = (e2.locations[0].offset, e2.locations[0].length)
    return tuple(sorted([e1_key, e2_key]))


_aio_priority_dict = {
    "IAO:0000305": 1,  # Title
    "IAO:0000315": 1,  # Abstract
    "IAO:0000318": 1,
    "IAO:0000615": 1,
    "IAO:0000319": 2,
    "IAO:0000317": 3,
    "IAO:0000633": 3,
    "IAO:0000316": 4,
}

_aio_doc_part_dict = {
    "IAO:0000305": "Title",  # Title
    "IAO:0000315": "Abstract",  # Abstract
    "IAO:0000318": "Results",
    "IAO:0000615": "Conclusion",
    "IAO:0000319": "Discussion",
    "IAO:0000317": "Methods",
    "IAO:0000633": "Materials",
    "IAO:0000316": "Introduction",
}


for file in os.listdir(input_path):
    _abs_file_path = os.path.join(input_path, file)
    instance = recode.parse_json_instance(_abs_file_path)
    candidate_generator.inference_with_instance(instance)
    res_list = []
    for document in instance.documents:
        for passage_idx, passage in enumerate(document.passages):
            for sentence_idx, sentence in enumerate(passage.sentences):
                if sentence.relations != None and len(sentence.relations) > 0:

                    seen_rel_set = set()

                    for rel in sentence.relations:
                        refid1 = rel.nodes[0].refid
                        refid2 = rel.nodes[1].refid
                        for annotation in sentence.annotations:
                            if annotation.id == refid1:
                                e1_anno = annotation

                            if annotation.id == refid2:
                                e2_anno = annotation

                        # e1_anno.locations[0].offset, e1_anno.locations[0].length, e2_anno.locations[0].offset, e2_anno.locations[0].length
                        _key = make_key(e1_anno, e2_anno)
                        if _key in seen_rel_set:
                            continue

                        seen_rel_set.add(make_key(e1_anno, e2_anno))
                        res_list.append(
                            {
                                "passage_idx": passage_idx,
                                "passage_offset": passage.offset,
                                "passage_infons": passage.infons,
                                "sentence_idx": sentence_idx,
                                "sen_offset": sentence.offset,
                                "sen_text": sentence.text,
                                "rel_id": rel.id,
                                "relation": sentence.relations[0].infons.type,
                                "e1_refid": refid1,
                                "e1_type": e1_anno.infons.type,
                                "e1_text": e1_anno.text,
                                "e1_loc_offset": e1_anno.locations[0].offset,
                                "e1_loc_length": e1_anno.locations[0].length,
                                "e2_refid": refid2,
                                "e2_type": e2_anno.infons.type,
                                "e2_text": e2_anno.text,
                                "e2_loc_offset": e2_anno.locations[0].offset,
                                "e2_loc_length": e2_anno.locations[0].length,
                                "passage_aio_priority": _aio_priority_dict.get(
                                    passage.infons.get("iao_id_0", ""), 99
                                ),
                                "passage_aio_part": _aio_doc_part_dict.get(
                                    passage.infons.get("iao_id_0", ""), "Unknown"
                                ),
                            }
                        )
    res_df = pd.DataFrame(res_list)
    os.makedirs(args.output_dir, exist_ok=True)
    _file_name = file.split(".")[0]
    res_df.to_csv(f"{args.output_dir}/{_file_name}.tsv", sep="\t")
