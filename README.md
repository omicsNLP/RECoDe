[![CoDiet](https://img.shields.io/badge/%F0%9F%8D%8E_a_CoDiet_study-5AA764)](https://www.codiet.eu)
[![DOI:10.64898/2026.03.03.709244](http://img.shields.io/badge/preprint_DOI-10.64898/2026.03.03.709244-BE2536.svg)](https://doi.org/10.64898/2026.03.03.709244)
[![DOI:10.5281/zenodo.19050553](http://img.shields.io/badge/data_DOI-10.5281/zenodo.19050553-3382C4.svg)](https://doi.org/10.5281/zenodo.19050553)

![alt text](https://github.com/omicsNLP/RECoDe/blob/main/RECoDe_CoDiet_Logo.png?raw=true)
A repository for "RECoDe - Relation Extraction for Diet, Non-Communicable Disease and Biomarker Associations: A CoDiet study" https://www.biorxiv.org/content/10.64898/2026.03.03.709244v1

## Prerequisite
```bash
# Create and activate conda environment
conda create -n recode python=3.11 -y
conda activate recode

# Install dependencies and the project
pip install .
```

## Running the RE Evaluation with LLMs

We support OpenAI-compatible clients to run our pipeline.
You can use OpenAI APIs or your own local models via a server (e.g., gpt-oss-20b). For example, see: [vllm article](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html).

### Inference

Run relation extraction on a dataset split:
```bash
python scripts/exp/run.py \
    --data_path ./data/annotation \
    --split test \
    --base_url http://localhost:8010/v1 \
    --model_name openai/gpt-oss-20b \
    --output_dir ./results
```

This produces two TSV files in `--output_dir`:
- `{split}_{model}_result.tsv` — predictions only (`type` column), used for evaluation
- `{split}_{model}_full.tsv` — gold + predictions for reference

### Evaluation

Evaluate predictions against a gold JSONL file:
```bash
python scripts/exp/evaluate.py \
    --gold ./data/annotation/test.jsonl \
    --pred ./results/test_model_result.tsv

# Or evaluate all TSV files in a directory:
python scripts/exp/evaluate.py \
    --gold ./data/annotation/test.jsonl \
    --pred_dir ./results/
```

This computes:
- **Multiclass**: accuracy, micro/macro/weighted precision/recall/F1, confusion matrix
- **Binary** (association vs NoAssociation): accuracy, binary/micro/macro/weighted metrics

## Full Pipeline (Candidate Generation → Filter → Inference → CoCoS)

The unified pipeline script [`scripts/pipeline.py`](./scripts/pipeline.py) runs the full workflow:

### 1. Generate Candidates
Extract relation candidate pairs from BioC JSON files (with NER annotations).
```bash
python scripts/pipeline.py candidate \
    --input_dir ./data/extraction/input \
    --output ./output/candidates.csv
```

### 2. Filter Candidates
Filter candidate pairs by entity type combinations.
```bash
python scripts/pipeline.py filter \
    --input ./output/candidates.csv \
    --output ./output/filtered.csv \
    --entity_type_filters default
```

Available filters (comma-separated):
| Filter | Entity pairs |
|--------|-------------|
| `food_disease` | foodRelated → diseasePhenotype |
| `food_bio` | foodRelated → geneSNP/proteinEnzyme/metabolite/microbiome |
| `disease_bio` | diseasePhenotype → geneSNP/proteinEnzyme/metabolite/microbiome |
| `food_food` | foodRelated → foodRelated |
| `bio_cross` | bio type → different bio type |
| `bio_self` | bio type → same bio type |
| `default` | all of the above |

### 3. Run Inference
Predict relation types for each candidate pair using an OpenAI-compatible LLM API.
```bash
python scripts/pipeline.py inference \
    --input ./output/filtered.csv \
    --output ./output/inference.csv \
    --base_url http://localhost:8010/v1 \
    --model_name openai/gpt-oss-20b

# For testing without an LLM (random predictions):
python scripts/pipeline.py inference \
    --input ./output/filtered.csv \
    --output ./output/inference.csv \
    --dummy
```

### 4. Build CoCoS
Build the Corpus-level Concept Summary (CoCoS) knowledge graph from inference results.

This step includes:

**Within each document (intra-doc):**
- Abbreviation expansion
- UK/US English normalization
- Entity normalization (cluster mentions sharing annotation IDs, pick representative text by frequency)
- Self-relation removal (token overlap)
- Hierarchical voting per entity pair → one relation label per (document, e1, e2)

**Across documents (inter-doc):**
- Aggregate document-level relations per entity pair
- Compute two scores per pair:
  - **Association Support (AS)**: `AS = N_assoc / (N_assoc + N_no)` — evidence for an association in general
  - **Effect Estimate (EE)**: `EE = (N_pos - N_neg) / N_assoc` — direction of the association (+1 = direct, -1 = inverse)
- Build knowledge graph (nodes with entity type/color/doc_cnt, edges with as_score/ee_score/doc_count)

```bash
python scripts/pipeline.py cocos \
    --input ./output/inference.csv \
    --input_dir ./data/extraction/input \
    --output_dir ./output/cocos \
    --eng_us_path ./data/extraction/resources/eng_us_uk.txt
```

Output files:
- `recode_cocos.graphml` — NetworkX graph (nodes: entity type, color, doc_cnt; edges: as_score, ee_score, counts)
- `recode_cocos.csv` — aggregated pair scores (as_score, ee_score, doc_count, relation type counts)
- `processed_relations.csv` — all relations after normalization

### Run All Steps at Once
```bash
python scripts/pipeline.py all \
    --input_dir ./data/extraction/input \
    --output_dir ./output \
    --base_url http://localhost:8010/v1 \
    --model_name openai/gpt-oss-20b \
    --eng_us_path ./data/extraction/resources/eng_us_uk.txt \
    --entity_type_filters default

# Full pipeline with dummy inference (for testing):
python scripts/pipeline.py all \
    --input_dir ./data/extraction/input \
    --output_dir ./output \
    --entity_type_filters food_disease \
    --dummy
```

## Creating Your Own Dataset

1. Prepare your data in BioC JSON format. See [this input example](./data/extraction/input/PMC7271748.json).
   - NER annotations must be included. See [the CoDiet Corpus paper](https://www.biorxiv.org/content/10.1101/2025.09.04.673740v1).
   - Example: [the CoDiet Electrum Corpus](https://zenodo.org/records/17610205)

2. Generate candidate relation pairs:
```bash
python scripts/pipeline.py candidate \
    --input_dir ./data/extraction/input \
    --output ./output/candidates.csv
```
   - Example output: [example_PMC7271748.tsv](./data/extraction/output/example_PMC7271748.tsv)

3. Filter, run inference, and build CoCoS (see above).
