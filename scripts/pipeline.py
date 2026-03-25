#!/usr/bin/env python
"""
RECoDe Pipeline
===============
Unified script for relation extraction from biomedical literature.

Steps:
  candidate  - Generate relation candidate pairs from BioC JSON files
  inference  - Predict relation types using LLM via OpenAI API
  cocos      - Build CoCoS (Corpus-level Concept Summary) from predictions
  all        - Run all steps sequentially

Usage:
  python pipeline.py candidate --input_dir ./data/extraction/input --output ./output/candidates.csv
  python pipeline.py inference --input ./output/candidates.csv --output ./output/inference.csv --base_url http://localhost:8010/v1 --model_name gpt-4
  python pipeline.py cocos --input ./output/inference.csv --output_dir ./output/cocos --eng_us_path ./data/extraction/resources/eng_us_uk.txt
  python pipeline.py all --input_dir ./data/extraction/input --output_dir ./output --base_url http://localhost:8010/v1 --model_name gpt-4
"""

import argparse
import os
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd
from tqdm import tqdm

import recode

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

TARGET_TUPLE_TYPES = [
    # food to disease
    ["foodRelated", "diseasePhenotype"],
    # food to bio
    ["foodRelated", "geneSNP"],
    ["foodRelated", "proteinEnzyme"],
    ["foodRelated", "metabolite"],
    ["foodRelated", "microbiome"],
    # disease to bio
    ["diseasePhenotype", "geneSNP"],
    ["diseasePhenotype", "proteinEnzyme"],
    ["diseasePhenotype", "metabolite"],
    ["diseasePhenotype", "microbiome"],
    # bio to bio
    ["geneSNP", "proteinEnzyme"],
    ["geneSNP", "metabolite"],
    ["geneSNP", "microbiome"],
    ["proteinEnzyme", "metabolite"],
    ["proteinEnzyme", "microbiome"],
    ["metabolite", "microbiome"],
    # self-relations
    ["foodRelated", "foodRelated"],
    ["diseasePhenotype", "diseasePhenotype"],
    ["geneSNP", "geneSNP"],
    ["proteinEnzyme", "proteinEnzyme"],
    ["metabolite", "metabolite"],
    ["microbiome", "microbiome"],
]

AIO_PRIORITY = {
    "IAO:0000318": 1,  # Results
    "IAO:0000615": 1,  # Conclusion
    "IAO:0000305": 2,  # Title
    "IAO:0000315": 2,  # Abstract
    "IAO:0000319": 3,  # Discussion
    "IAO:0000317": 4,  # Methods
    "IAO:0000633": 4,  # Materials
    "IAO:0000316": 5,  # Introduction
    "IAO:0000630": 99, # Keywords
    "IAO:0000314": 99, # Document Part
}

AIO_DOC_PART = {
    "IAO:0000305": "Title",
    "IAO:0000315": "Abstract",
    "IAO:0000318": "Results",
    "IAO:0000615": "Conclusion",
    "IAO:0000319": "Discussion",
    "IAO:0000317": "Methods",
    "IAO:0000633": "Materials",
    "IAO:0000316": "Introduction",
}

POSITIVE_RELATIONS = ["increaseAssociation", "positiveCorrelation", "consists"]
NEGATIVE_RELATIONS = ["decreaseAssociation", "negativeCorrelation", "substitution"]
NEUTRAL_RELATIONS = ["causalEffect", "association"]
NO_RELATIONS = ["NoAssociation", "Unrelated"]

# Entity type filter presets
ENTITY_TYPE_FILTERS = {
    "food_disease": [
        ("foodRelated", "diseasePhenotype"),
    ],
    "food_bio": [
        ("foodRelated", "geneSNP"),
        ("foodRelated", "proteinEnzyme"),
        ("foodRelated", "metabolite"),
        ("foodRelated", "microbiome"),
    ],
    "disease_bio": [
        ("diseasePhenotype", "geneSNP"),
        ("diseasePhenotype", "proteinEnzyme"),
        ("diseasePhenotype", "metabolite"),
        ("diseasePhenotype", "microbiome"),
    ],
    "food_food": [
        ("foodRelated", "foodRelated"),
    ],
    "bio_cross": [
        ("geneSNP", "proteinEnzyme"),
        ("geneSNP", "metabolite"),
        ("geneSNP", "microbiome"),
        ("proteinEnzyme", "metabolite"),
        ("proteinEnzyme", "microbiome"),
        ("metabolite", "microbiome"),
    ],
    "bio_self": [
        ("diseasePhenotype", "diseasePhenotype"),
        ("geneSNP", "geneSNP"),
        ("proteinEnzyme", "proteinEnzyme"),
        ("metabolite", "metabolite"),
        ("microbiome", "microbiome"),
    ],
}


def resolve_entity_type_filters(filter_str):
    """Resolve --entity_type_filters string to a set of (e1_type, e2_type) tuples.

    Examples:
        "default"                        -> all types
        "food_disease,food_bio"          -> food_disease + food_bio
        "food_disease,disease_bio,food_food" -> those three
    """
    if filter_str is None or filter_str == "default":
        pairs = set()
        for v in ENTITY_TYPE_FILTERS.values():
            pairs.update(v)
        return pairs

    pairs = set()
    for name in filter_str.split(","):
        name = name.strip()
        if name not in ENTITY_TYPE_FILTERS:
            raise ValueError(
                f"Unknown entity_type_filter: '{name}'. "
                f"Available: {list(ENTITY_TYPE_FILTERS.keys())}"
            )
        pairs.update(ENTITY_TYPE_FILTERS[name])
    return pairs


def apply_entity_type_filter(df, filter_pairs):
    """Filter DataFrame to only include rows matching the given (e1_type, e2_type) pairs."""
    mask = df.apply(
        lambda row: (row["e1_type"], row["e2_type"]) in filter_pairs, axis=1
    )
    filtered = df[mask].copy()
    print(f"Entity type filter: {len(df)} → {len(filtered)} rows")
    return filtered


# ═══════════════════════════════════════════════════════════════
# Step 1: Generate Candidates
# ═══════════════════════════════════════════════════════════════

def _make_key(e1, e2):
    e1_key = (e1.locations[0].offset, e1.locations[0].length)
    e2_key = (e2.locations[0].offset, e2.locations[0].length)
    return tuple(sorted([e1_key, e2_key]))


def generate_candidates(input_dir, output_path, max_text_len=1000):
    """Generate relation candidate pairs from BioC JSON files."""
    candidate_generator = recode.CoDietRelationCandidateGenerator(TARGET_TUPLE_TYPES)
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    all_rows = []

    for file in tqdm(files, desc="Generating candidates"):
        abs_path = os.path.join(input_dir, file)
        instance = recode.parse_json_instance(abs_path)
        candidate_generator.inference_with_instance(instance)
        pmcid = file.split(".")[0]

        # Build annotation ID and annotator mappings for this file
        anno_id_map = {}
        anno_annotator_map = {}
        for document in instance.documents:
            for passage in document.passages:
                for sentence in passage.sentences:
                    for annotation in sentence.annotations:
                        key = f"{file}_{annotation.id}"
                        anno_id_map[key] = getattr(annotation.infons, "identifier", "")
                        anno_annotator_map[key] = getattr(annotation.infons, "annotator", "")

        for document in instance.documents:
            for passage_idx, passage in enumerate(document.passages):
                for sentence_idx, sentence in enumerate(passage.sentences):
                    if not sentence.relations or len(sentence.relations) == 0:
                        continue

                    seen = set()
                    for rel in sentence.relations:
                        refid1 = rel.nodes[0].refid
                        refid2 = rel.nodes[1].refid

                        e1_anno = e2_anno = None
                        for annotation in sentence.annotations:
                            if annotation.id == refid1:
                                e1_anno = annotation
                            if annotation.id == refid2:
                                e2_anno = annotation

                        if e1_anno is None or e2_anno is None:
                            continue

                        # Skip entity types containing underscore
                        if "_" in e1_anno.infons.type or "_" in e2_anno.infons.type:
                            continue

                        key = _make_key(e1_anno, e2_anno)
                        if key in seen:
                            continue
                        seen.add(key)

                        e1_key = f"{file}_{refid1}"
                        e2_key = f"{file}_{refid2}"

                        row = {
                            "PMCID": pmcid,
                            "passage_idx": passage_idx,
                            "sentence_idx": sentence_idx,
                            "sen_offset": sentence.offset,
                            "sen_text": sentence.text,
                            "rel_id": rel.id,
                            "relation": rel.infons.type,
                            "e1_refid": refid1,
                            "e1_type": e1_anno.infons.type,
                            "e1_text": e1_anno.text,
                            "e1_loc_offset": e1_anno.locations[0].offset,
                            "e1_loc_length": e1_anno.locations[0].length,
                            "e1_annotation_id": anno_id_map.get(e1_key, ""),
                            "e1_annotator": anno_annotator_map.get(e1_key, ""),
                            "e2_refid": refid2,
                            "e2_type": e2_anno.infons.type,
                            "e2_text": e2_anno.text,
                            "e2_loc_offset": e2_anno.locations[0].offset,
                            "e2_loc_length": e2_anno.locations[0].length,
                            "e2_annotation_id": anno_id_map.get(e2_key, ""),
                            "e2_annotator": anno_annotator_map.get(e2_key, ""),
                            "passage_aio_priority": AIO_PRIORITY.get(
                                passage.infons.get("iao_id_1", "") or passage.infons.get("iao_id_2", ""), 99
                            ),
                            "passage_aio_part": AIO_DOC_PART.get(
                                passage.infons.get("iao_id_1", "") or passage.infons.get("iao_id_2", ""), "Unknown"
                            ),
                        }
                        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        print("No candidates generated.")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    if max_text_len:
        df = df[df["sen_text"].str.len() <= max_text_len]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} candidates → {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════
# Step 2: Filter
# ═══════════════════════════════════════════════════════════════

def filter_candidates(input_path, output_path, entity_type_filters="default"):
    """Filter candidate pairs by entity type combinations."""
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} candidates from {input_path}")

    filter_pairs = resolve_entity_type_filters(entity_type_filters)
    df = apply_entity_type_filter(df, filter_pairs)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Filtered candidates → {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════
# Step 3: Inference
# ═══════════════════════════════════════════════════════════════

def _dummy_predict(*args, **kwargs):
    """Random prediction for testing (no LLM needed)."""
    import random
    return random.choice(recode.labels)


def run_inference(input_path, output_path, base_url=None, model_name=None, api_key=None,
                  start_idx=0, end_idx=None, temperature=0.2, top_p=0.8,
                  num_max_tokens=512, num_trials=3, dummy=False):
    """Run relation prediction on candidate pairs via OpenAI API."""
    df = pd.read_csv(input_path)

    if end_idx is None:
        end_idx = len(df) - 1

    cdf = df.iloc[start_idx:end_idx + 1].copy()

    cdf["transformed_text"] = cdf.apply(
        lambda row: recode.read._get_transformed_text_(
            row["sen_text"],
            int(row["e1_loc_offset"] - row["sen_offset"]),
            int(row["e1_loc_length"]),
            int(row["e2_loc_offset"] - row["sen_offset"]),
            int(row["e2_loc_length"]),
        ),
        axis=1,
    )

    predict_fn = _dummy_predict if dummy else recode.predict

    results = []
    for idx, row in tqdm(cdf.iterrows(), total=len(cdf), desc="Running inference"):
        result = predict_fn(
            row["e1_text"],
            row["e2_text"],
            row["transformed_text"],
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            num_max_tokens=num_max_tokens,
            num_trials=num_trials,
        )
        results.append(result)
        print(f"  [{idx}] {row['transformed_text']}")
        print(f"         → {result}")
        print()

    cdf["recode_result"] = results

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cdf.to_csv(output_path, index=False)
    print(f"Inference complete ({len(cdf)} rows) → {output_path}")
    return cdf


# ═══════════════════════════════════════════════════════════════
# Step 4: Build CoCoS
# ═══════════════════════════════════════════════════════════════

# --- Abbreviation helpers ---

def _calculate_abbr_score(long_form, is_hybrid=False):
    if is_hybrid:
        return long_form.score if long_form.score is not None else 0
    algos = long_form.extraction_algorithms or []
    if "fulltext" in algos and "HybridDK+" in algos:
        return 2000
    elif "fulltext" in algos:
        return 1000
    elif "HybridDK+" in algos:
        return None
    return long_form.score if long_form.score is not None else 0


def _merge_abbreviations(instance):
    merged = {}
    if instance.infons.abbreviations:
        for abbr in instance.infons.abbreviations:
            for lf in abbr.long_forms:
                score = _calculate_abbr_score(lf)
                if score is None or score < 4.0:
                    continue
                key = abbr.short_form
                if key not in merged:
                    merged[key] = {}
                merged[key][lf.text.lower()] = score

    if instance.infons.hybrid_abbreviations:
        for abbr in instance.infons.hybrid_abbreviations:
            for lf in abbr.long_forms:
                score = _calculate_abbr_score(lf, is_hybrid=True)
                if score is None or score < 4.0:
                    continue
                key = abbr.short_form
                if key not in merged:
                    merged[key] = {}
                merged[key][lf.text.lower()] = score

    return merged


def load_abbreviations(abbr_dir):
    """Load abbreviation mappings from abbr JSON files."""
    abbr_dict = {}

    for file in tqdm(os.listdir(abbr_dir), desc="Loading abbreviations"):
        try:
            instance = recode.parse_json_instance(os.path.join(abbr_dir, file))
            pmc_id = instance.infons.pmcid
            merged = _merge_abbreviations(instance)

            for short_form, long_forms in merged.items():
                if short_form not in abbr_dict:
                    abbr_dict[short_form] = {}
                if pmc_id not in abbr_dict[short_form]:
                    abbr_dict[short_form][pmc_id] = {}
                abbr_dict[short_form][pmc_id].update(long_forms)
        except Exception:
            continue

    return abbr_dict


def expand_abbreviations(df, abbr_dict):
    """Expand abbreviations in entity strings using per-PMCID mappings."""
    def _expand(target, pmc_id):
        new_target = target
        for abbr_term, pmcid_dict in abbr_dict.items():
            if target.startswith(abbr_term) or f" {abbr_term}" in target:
                if pmc_id in pmcid_dict:
                    score_dict = pmcid_dict[pmc_id]
                    best_long = max(score_dict, key=score_dict.get)
                    new_target = new_target.replace(abbr_term, best_long)
        return new_target

    df["e1_str"] = df.apply(lambda r: _expand(str(r["e1_text"]), r["PMCID"]), axis=1)
    df["e2_str"] = df.apply(lambda r: _expand(str(r["e2_text"]), r["PMCID"]), axis=1)
    return df


# --- Normalization helpers (annotation_id string clustering, from 2_cocos_score) ---

def _normalize_aids_str(x):
    """Normalize annotation_id string: strip whitespace, join with comma."""
    if pd.isna(x):
        return pd.NA
    toks = [t.strip() for t in str(x).split(",") if t.strip()]
    if not toks:
        return pd.NA
    return ",".join(toks)


def _get_aid_to_rep(cdf):
    """Cluster annotation IDs within a document by greedy bridge-merge (frequency-ordered).

    Returns:
        aid_to_rep_aid_str: dict mapping each individual aid → representative aid string
        rep_aids_str_to_all_aid_set: dict mapping representative → set of all member aids
    """
    merged_cnt = Counter()
    aid_to_rep_aid_str = {}
    rep_aids_str_to_all_aid_set = {}

    for col in ["e1_annotation_id", "e2_annotation_id"]:
        merged_cnt.update(cdf[col].dropna().astype(str))

    def ensure_rep(rep):
        if rep not in rep_aids_str_to_all_aid_set:
            rep_aids_str_to_all_aid_set[rep] = set()

    def merge_reps(rep_keep, rep_drop):
        if rep_keep == rep_drop:
            return
        ensure_rep(rep_keep)
        moved = rep_aids_str_to_all_aid_set.pop(rep_drop, set())
        rep_aids_str_to_all_aid_set[rep_keep].update(moved)
        for a in moved:
            aid_to_rep_aid_str[a] = rep_keep

    for aids_str, cnt in merged_cnt.most_common():
        aids = [a.strip() for a in aids_str.split(",") if a.strip()]
        if not aids:
            continue

        existing_reps = {aid_to_rep_aid_str[a] for a in aids if a in aid_to_rep_aid_str}
        if len(existing_reps) >= 2:
            reps_sorted = sorted(existing_reps, key=lambda r: (-merged_cnt.get(r, 0), r))
            rep_keep = reps_sorted[0]
            for rep_drop in reps_sorted[1:]:
                merge_reps(rep_keep, rep_drop)
            existing_reps = {rep_keep}

        if len(existing_reps) == 1:
            rep = next(iter(existing_reps))
        else:
            rep = aids_str

        ensure_rep(rep)
        rep_aids_str_to_all_aid_set[rep].update(aids)
        for a in aids:
            aid_to_rep_aid_str[a] = rep

    return aid_to_rep_aid_str, rep_aids_str_to_all_aid_set


def _add_rep_aid_cols(cdf, aid_to_rep_aid_str, rep_aids_str_to_all_aid_set):
    """Add representative annotation_id columns to the DataFrame."""
    cdf = cdf.copy()

    e1_norm = cdf["e1_annotation_id"].map(_normalize_aids_str)
    e2_norm = cdf["e2_annotation_id"].map(_normalize_aids_str)

    def to_rep(normed_aids_str):
        if pd.isna(normed_aids_str):
            return pd.NA
        for t in str(normed_aids_str).split(","):
            t = t.strip()
            if not t:
                continue
            rep = aid_to_rep_aid_str.get(t)
            if rep is not None:
                return rep
        return normed_aids_str

    e1_rep = e1_norm.map(to_rep)
    e2_rep = e2_norm.map(to_rep)

    e1_rep_all = e1_rep.map(rep_aids_str_to_all_aid_set).combine_first(e1_rep)
    e2_rep_all = e2_rep.map(rep_aids_str_to_all_aid_set).combine_first(e2_rep)

    cdf["e1_rep_aid"] = e1_rep
    cdf["e2_rep_aid"] = e2_rep
    cdf["e1_rep_aid_to_all"] = e1_rep_all
    cdf["e2_rep_aid_to_all"] = e2_rep_all

    return cdf


def _normalize_entities_per_doc(df):
    """Normalize entities per document using annotation_id string clustering.

    For each PMCID:
    1. Cluster annotation IDs by greedy bridge-merge (frequency-ordered)
    2. Map each mention to its cluster representative
    3. Derive top mention text and top entity type per cluster (corpus-wide)

    Adds columns: e1_str_norm, e2_str_norm, norm_e1_type, norm_e2_type
    """
    # Step 1: Per-document clustering
    print("  Clustering annotation IDs per document...")
    processed_parts = []
    for pmcid in tqdm(df["PMCID"].unique(), desc="Normalizing entities"):
        doc_df = df[df["PMCID"] == pmcid]
        aid_to_rep, rep_to_all = _get_aid_to_rep(doc_df)
        doc_df = _add_rep_aid_cols(doc_df, aid_to_rep, rep_to_all)
        processed_parts.append(doc_df)

    merged = pd.concat(processed_parts, ignore_index=True)

    # Step 2: Convert rep_aid_to_all sets to sorted strings for grouping
    merged["e1_rep_aid_to_all_str"] = merged["e1_rep_aid_to_all"].map(
        lambda s: ",".join(sorted(s)) if isinstance(s, set) else pd.NA
    )
    merged["e2_rep_aid_to_all_str"] = merged["e2_rep_aid_to_all"].map(
        lambda s: ",".join(sorted(s)) if isinstance(s, set) else pd.NA
    )

    # Step 3: Top mention text per cluster (corpus-wide)
    long_cdf = pd.concat([
        merged[["e1_rep_aid_to_all_str", "e1_text"]].rename(
            columns={"e1_rep_aid_to_all_str": "k", "e1_text": "text"}
        ),
        merged[["e2_rep_aid_to_all_str", "e2_text"]].rename(
            columns={"e2_rep_aid_to_all_str": "k", "e2_text": "text"}
        ),
    ], ignore_index=True).dropna(subset=["k", "text"])

    kt_cnt = long_cdf.groupby(["k", "text"]).size()
    aid_all_to_top_mention = kt_cnt.groupby(level=0).idxmax().map(lambda x: x[1]).to_dict()

    merged["e1_top_mention"] = merged["e1_rep_aid_to_all_str"].map(aid_all_to_top_mention)
    merged["e2_top_mention"] = merged["e2_rep_aid_to_all_str"].map(aid_all_to_top_mention)

    # Step 4: Top entity type per mention (corpus-wide majority vote)
    long_mt = pd.concat([
        merged[["e1_top_mention", "e1_type"]].rename(
            columns={"e1_top_mention": "mention", "e1_type": "type"}
        ),
        merged[["e2_top_mention", "e2_type"]].rename(
            columns={"e2_top_mention": "mention", "e2_type": "type"}
        ),
    ], ignore_index=True).dropna(subset=["mention", "type"])

    mt_cnt = long_mt.groupby(["mention", "type"]).size()
    top_mention_to_top_type = mt_cnt.groupby(level=0).idxmax().map(lambda x: x[1]).to_dict()

    merged["norm_e1_type"] = merged["e1_top_mention"].map(top_mention_to_top_type)
    merged["norm_e2_type"] = merged["e2_top_mention"].map(top_mention_to_top_type)

    # Step 5: Top annotation_id per mention
    long_ma = pd.concat([
        merged[["e1_top_mention", "e1_annotation_id"]].rename(
            columns={"e1_top_mention": "mention", "e1_annotation_id": "annotation_id"}
        ),
        merged[["e2_top_mention", "e2_annotation_id"]].rename(
            columns={"e2_top_mention": "mention", "e2_annotation_id": "annotation_id"}
        ),
    ], ignore_index=True).dropna(subset=["mention", "annotation_id"])

    ma_cnt = long_ma.groupby(["mention", "annotation_id"]).size()
    top_mention_to_top_ann = ma_cnt.groupby(level=0).idxmax().map(lambda x: x[1]).to_dict()

    merged["norm_e1_id"] = merged["e1_top_mention"].map(top_mention_to_top_ann)
    merged["norm_e2_id"] = merged["e2_top_mention"].map(top_mention_to_top_ann)

    # Set normalized surface forms
    merged["e1_str_norm"] = merged["e1_top_mention"]
    merged["e2_str_norm"] = merged["e2_top_mention"]

    return merged


def load_uk_us_dict(path):
    """Load UK US English dictionary (tab-separated)."""
    uk_us = {}
    for line in open(path):
        parts = line.strip().split("\t")
        if len(parts) == 2:
            uk_us[parts[0]] = parts[1]
    return uk_us



def _has_token_overlap(str1, str2):
    """Check if two entity strings share tokens (self-relation detection)."""
    def get_variants(s):
        tokens = set(s.lower().split())
        variants = set()
        for t in tokens:
            variants.add(t)
            variants.add(t.rstrip("s"))
            if not t.endswith("s"):
                variants.add(t + "s")
        return variants

    return bool(get_variants(str1) & get_variants(str2))


def postprocess_relations(df):
    """Pass through relation labels (no keyword heuristic applied)."""
    df["processed_relation"] = df["recode_result"]
    return df


# --- Aggregation ---

def _aggregate_relation_within_doc(
    recode_results,
    no_assoc_label="NoAssociation",
    tie_prefer="association",
    ratio_2x=2.0,
):
    """Aggregate multiple sentence-level relation labels for ONE (doc, e1, e2) pair.

    Hierarchical voting:
    1) Remove Unrelated (already filtered before calling)
    2) NoAssociation vs any association → majority wins
    3) Among associations: correlation vs association vs consist/substitution → highest group
    4) Within group: directional resolution with 2x threshold
    """
    labels = [str(x).strip() for x in recode_results]
    cnt = Counter(labels)

    corr_pos = cnt.get("positiveCorrelation", 0)
    corr_neg = cnt.get("negativeCorrelation", 0)
    corr_total = corr_pos + corr_neg

    assoc_neutral = cnt.get("association", 0)
    assoc_inc = cnt.get("increaseAssociation", 0)
    assoc_dec = cnt.get("decreaseAssociation", 0)
    causal = cnt.get("causalEffect", 0)
    assoc_total = assoc_neutral + assoc_inc + assoc_dec + causal

    consist = cnt.get("consists", 0)
    substitution = cnt.get("substitution", 0)
    consist_sub_total = consist + substitution

    any_assoc_total = corr_total + assoc_total + consist_sub_total

    # Step 2: NoAssociation vs any association
    no_cnt = cnt.get(no_assoc_label, 0)
    if no_cnt > any_assoc_total:
        return no_assoc_label
    if any_assoc_total == 0:
        return no_assoc_label

    # Step 3: pick winning group (tie-break: correlation > association > consist/sub)
    group_scores = {
        "correlation": corr_total,
        "association": assoc_total,
        "consist_or_substitution": consist_sub_total,
    }
    winning_group = max(
        group_scores,
        key=lambda g: (group_scores[g], {"correlation": 3, "association": 2, "consist_or_substitution": 1}[g]),
    )

    # Step 4: correlation resolution
    if winning_group == "correlation":
        if corr_pos and not corr_neg:
            return "positiveCorrelation"
        if corr_neg and not corr_pos:
            return "negativeCorrelation"
        if corr_pos >= ratio_2x * corr_neg:
            return "positiveCorrelation"
        if corr_neg >= ratio_2x * corr_pos:
            return "negativeCorrelation"
        return tie_prefer

    # Step 5: association resolution
    if winning_group == "association":
        if causal and causal >= max(1, assoc_neutral + assoc_inc + assoc_dec):
            return "causalEffect"
        if assoc_inc and not assoc_dec:
            return "increaseAssociation"
        if assoc_dec and not assoc_inc:
            return "decreaseAssociation"
        if assoc_inc and assoc_dec:
            if assoc_inc >= ratio_2x * assoc_dec:
                return "increaseAssociation"
            if assoc_dec >= ratio_2x * assoc_inc:
                return "decreaseAssociation"
            return "association"
        return "association"

    # Step 6: consist/substitution resolution
    if consist >= substitution:
        return "consists"
    return "substitution"


def aggregate_relations(df):
    """Aggregate relations at document level and compute edge metrics."""
    group_cols = ["e1_str_norm", "e2_str_norm", "PMCID"]

    # Step 1: filter out Unrelated
    df = df[df["processed_relation"] != "Unrelated"].copy()

    # Doc-level: hierarchical voting per (e1, e2, PMCID)
    doc_level = []
    for (e1, e2, pmcid), group in df.groupby(group_cols):
        dominant = _aggregate_relation_within_doc(group["processed_relation"])
        doc_level.append({
            "e1_str_norm": e1,
            "e2_str_norm": e2,
            "PMCID": pmcid,
            "doc_relation": dominant,
        })

    doc_df = pd.DataFrame(doc_level)

    combo_rows = []
    for (e1, e2), group in doc_df.groupby(["e1_str_norm", "e2_str_norm"]):
        rels = group["doc_relation"]
        pos = int(rels.isin(POSITIVE_RELATIONS).sum())
        neg = int(rels.isin(NEGATIVE_RELATIONS).sum())
        neutral = int(rels.isin(NEUTRAL_RELATIONS).sum())
        no_assoc = int(rels.isin(NO_RELATIONS).sum())

        yes_total = pos + neg + neutral
        total = yes_total + no_assoc
        yes_ratio = yes_total / total if total > 0 else 0
        direction = (pos - neg) / yes_total if yes_total > 0 else 0

        rel_counter = Counter(rels)

        combo_rows.append({
            "e1_str_norm": e1,
            "e2_str_norm": e2,
            "doc_count": len(group),
            "yes_count": yes_total,
            "total_count": total,
            "pos_count": pos,
            "neg_count": neg,
            "neutral_count": neutral,
            "no_association_count": no_assoc,
            "as_score": round(yes_ratio, 4),
            "ee_score": round(direction, 4),
            **{rel: rel_counter.get(rel, 0) for rel in recode.labels},
            "pmcids": ",".join(group["PMCID"].unique()),
        })

    combo_df = pd.DataFrame(combo_rows)
    combo_df = combo_df.sort_values("doc_count", ascending=False)
    return combo_df


# --- Graph construction ---

CATEGORY_COLOR = {
    "foodRelated": "#a2fd25",
    "metabolite": "#eac6ff",
    "microbiome": "#dc3c00",
    "proteinEnzyme": "#bd5f00",
    "geneSNP": "#01af1e",
    "diseasePhenotype": "#b850ee",
}


def build_graph(combo_df, output_dir, node_meta=None, min_yes_count=1):
    """Construct NetworkX graph and export to GraphML + CSV."""
    filtered = combo_df[combo_df["yes_count"] >= min_yes_count].copy()
    if node_meta is None:
        node_meta = {}

    G = nx.Graph()

    for _, row in filtered.iterrows():
        e1 = row["e1_str_norm"]
        e2 = row["e2_str_norm"]

        for entity in [e1, e2]:
            if entity not in G:
                meta = node_meta.get(entity, {})
                etype = meta.get("type", "unknown")
                G.add_node(
                    entity,
                    type=etype,
                    color=CATEGORY_COLOR.get(etype, "#999999"),
                    doc_cnt=int(meta.get("doc_cnt", 0)),
                    annotation_id=str(meta.get("annotation_id", "")),
                )

        G.add_edge(
            e1, e2,
            as_score=float(row["as_score"]),
            ee_score=float(row["ee_score"]),
            doc_count=int(row["doc_count"]),
            yes_count=int(row["yes_count"]),
            total_count=int(row["total_count"]),
            pos_count=int(row["pos_count"]),
            neg_count=int(row["neg_count"]),
            neutral_count=int(row["neutral_count"]),
            no_association_count=int(row["no_association_count"]),
            pmcids=row["pmcids"],
        )

    os.makedirs(output_dir, exist_ok=True)
    graphml_path = os.path.join(output_dir, "recode_cocos.graphml")
    csv_path = os.path.join(output_dir, "recode_cocos.csv")

    nx.write_graphml(G, graphml_path)
    filtered.to_csv(csv_path, index=False)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  → {graphml_path}")
    print(f"  → {csv_path}")
    return G


# --- CoCoS main entry ---

def build_cocos(input_path, input_dir, output_dir, abbr_dir=None, eng_us_path=None,
                min_yes_count=1):
    """Build CoCoS from inference results."""
    print("Loading inference results...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["recode_result"])

    # --- Abbreviation expansion ---
    if abbr_dir and os.path.isdir(abbr_dir):
        print("Loading pre-generated abbreviations...")
        abbr_dict = load_abbreviations(abbr_dir)
        print(f"  {len(abbr_dict)} abbreviations loaded")
        df = expand_abbreviations(df, abbr_dict)
    elif input_dir:
        print("Generating abbreviations from input BioC JSONs...")
        abbr_cache = os.path.join(output_dir, "_abbr_cache")
        recode.AbbrExtractor(input_dir, abbr_cache).extract_abbr_and_save_to_dir()
        abbr_dict = load_abbreviations(abbr_cache)
        print(f"  {len(abbr_dict)} abbreviations loaded")
        df = expand_abbreviations(df, abbr_dict)
    else:
        df["e1_str"] = df["e1_text"]
        df["e2_str"] = df["e2_text"]

    # --- UK/US normalization ---
    uk_us_dict = {}
    if eng_us_path and os.path.isfile(eng_us_path):
        uk_us_dict = load_uk_us_dict(eng_us_path)
        print(f"  UK/US dict: {len(uk_us_dict)} entries")

    # --- Entity normalization (Union-Find per document) ---
    print("Normalizing entities (per-document Union-Find)...")
    df = _normalize_entities_per_doc(df)

    # --- Remove self-relations (token overlap) ---
    before = len(df)
    df = df[~df.apply(
        lambda r: _has_token_overlap(str(r["e1_str_norm"]), str(r["e2_str_norm"])),
        axis=1,
    )]
    print(f"  Removed {before - len(df)} self-relations (token overlap)")

    # --- Post-process relations ---
    print("Post-processing relations...")
    df = postprocess_relations(df)

    # --- Aggregate and build graph ---
    print("Aggregating relations...")
    combo_df = aggregate_relations(df)

    # Build node metadata (type, doc_cnt, annotation_id per entity)
    print("Building node metadata...")
    long_entities = pd.concat([
        df[["e1_str_norm", "norm_e1_type", "norm_e1_id", "PMCID"]].rename(
            columns={"e1_str_norm": "entity", "norm_e1_type": "type", "norm_e1_id": "ann_id"}
        ),
        df[["e2_str_norm", "norm_e2_type", "norm_e2_id", "PMCID"]].rename(
            columns={"e2_str_norm": "entity", "norm_e2_type": "type", "norm_e2_id": "ann_id"}
        ),
    ], ignore_index=True).dropna(subset=["entity"])

    entity_doc_cnt = long_entities.groupby("entity")["PMCID"].nunique()
    entity_top_type = (
        long_entities.groupby(["entity", "type"]).size()
        .groupby(level=0).idxmax().map(lambda x: x[1]).to_dict()
    )
    entity_top_ann = (
        long_entities.groupby(["entity", "ann_id"]).size()
        .groupby(level=0).idxmax().map(lambda x: x[1]).to_dict()
    )

    node_meta = {}
    for entity in set(combo_df["e1_str_norm"]) | set(combo_df["e2_str_norm"]):
        node_meta[entity] = {
            "type": entity_top_type.get(entity, "unknown"),
            "doc_cnt": int(entity_doc_cnt.get(entity, 0)),
            "annotation_id": str(entity_top_ann.get(entity, "")),
        }

    print("Building graph...")
    G = build_graph(combo_df, output_dir, node_meta=node_meta, min_yes_count=min_yes_count)

    # Save processed data
    processed_path = os.path.join(output_dir, "processed_relations.csv")
    df.to_csv(processed_path, index=False)
    print(f"  → {processed_path}")

    return G


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RECoDe Pipeline - Relation Extraction for Diet, NCD and Biomarker Associations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="step", required=True)

    # --- candidate ---
    p_cand = subparsers.add_parser("candidate", help="Generate candidate pairs from BioC JSON")
    p_cand.add_argument("--input_dir", required=True, help="Directory of BioC JSON files")
    p_cand.add_argument("--output", default="./output/candidates.csv", help="Output CSV path")
    p_cand.add_argument("--max_text_len", type=int, default=1000, help="Max sentence length filter")

    # --- filter ---
    p_filt = subparsers.add_parser("filter", help="Filter candidates by entity type combinations")
    p_filt.add_argument("--input", required=True, help="Candidates CSV path")
    p_filt.add_argument("--output", default="./output/filtered.csv", help="Filtered output CSV path")
    p_filt.add_argument("--entity_type_filters", type=str, default="default",
                        help="Entity type filters: 'default' (all), or comma-separated: "
                             "food_disease,food_bio,disease_bio,food_food,bio_cross,bio_self")

    # --- inference ---
    p_inf = subparsers.add_parser("inference", help="Run relation prediction via OpenAI API")
    p_inf.add_argument("--input", required=True, help="Filtered candidates CSV path")
    p_inf.add_argument("--output", default="./output/inference.csv", help="Output CSV path")
    p_inf.add_argument("--base_url", default=None, help="OpenAI-compatible API base URL")
    p_inf.add_argument("--model_name", default=None, help="Model name")
    p_inf.add_argument("--api_key", default="EMPTY", help="API key")
    p_inf.add_argument("--start_idx", type=int, default=0)
    p_inf.add_argument("--end_idx", type=int, default=None)
    p_inf.add_argument("--dummy", action="store_true", help="Use random predictions (no LLM, for testing)")

    # --- cocos ---
    p_cocos = subparsers.add_parser("cocos", help="Build CoCoS from inference results")
    p_cocos.add_argument("--input", required=True, help="Inference results CSV path")
    p_cocos.add_argument("--input_dir", default=None, help="BioC JSON dir (for abbreviation extraction)")
    p_cocos.add_argument("--output_dir", default="./output/cocos", help="Output directory")
    p_cocos.add_argument("--abbr_dir", default=None, help="Pre-generated abbreviation JSON dir")
    p_cocos.add_argument("--eng_us_path", default=None, help="UK/US English dictionary path")
    p_cocos.add_argument("--min_yes_count", type=int, default=1)

    # --- all ---
    p_all = subparsers.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--input_dir", required=True, help="Directory of BioC JSON files")
    p_all.add_argument("--output_dir", default="./output", help="Output directory")
    p_all.add_argument("--base_url", default=None, help="OpenAI-compatible API base URL")
    p_all.add_argument("--model_name", default=None, help="Model name")
    p_all.add_argument("--api_key", default="EMPTY", help="API key")
    p_all.add_argument("--abbr_dir", default=None, help="Pre-generated abbreviation JSON dir")
    p_all.add_argument("--eng_us_path", default=None, help="UK/US English dictionary path")
    p_all.add_argument("--start_idx", type=int, default=0)
    p_all.add_argument("--end_idx", type=int, default=None)
    p_all.add_argument("--entity_type_filters", type=str, default="default",
                       help="Entity type filters: 'default' (all), or comma-separated: "
                            "food_disease,food_bio,disease_bio,food_food,bio_cross,bio_self")
    p_all.add_argument("--dummy", action="store_true", help="Use random predictions (no LLM, for testing)")

    args = parser.parse_args()

    if args.step == "candidate":
        generate_candidates(args.input_dir, args.output, args.max_text_len)

    elif args.step == "filter":
        filter_candidates(args.input, args.output, args.entity_type_filters)

    elif args.step == "inference":
        run_inference(
            args.input, args.output, args.base_url, args.model_name,
            args.api_key, args.start_idx, args.end_idx,
            dummy=args.dummy,
        )

    elif args.step == "cocos":
        build_cocos(
            args.input, args.input_dir, args.output_dir,
            abbr_dir=args.abbr_dir, eng_us_path=args.eng_us_path,
            min_yes_count=args.min_yes_count,
        )

    elif args.step == "all":
        cand_path = os.path.join(args.output_dir, "candidates.csv")
        filt_path = os.path.join(args.output_dir, "filtered.csv")
        inf_path = os.path.join(args.output_dir, "inference.csv")
        cocos_dir = os.path.join(args.output_dir, "cocos")

        generate_candidates(args.input_dir, cand_path)
        filter_candidates(cand_path, filt_path, args.entity_type_filters)
        run_inference(
            filt_path, inf_path, args.base_url, args.model_name,
            args.api_key, args.start_idx, args.end_idx,
            dummy=args.dummy,
        )
        build_cocos(
            inf_path, args.input_dir, cocos_dir,
            abbr_dir=args.abbr_dir, eng_us_path=args.eng_us_path,
        )


if __name__ == "__main__":
    main()
