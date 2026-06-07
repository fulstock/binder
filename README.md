# BINDER — Russian fork

Russian-language fork of Microsoft's [BINDER (Bi-Encoder for Named Entity Recognition via Contrastive Learning)](https://openreview.net/forum?id=9EAQVEINuum) (ICLR 2023). Adapted for Russian NER and extended to additional datasets.

The original paper, model design, and citation are preserved at the bottom of this README. The sections above describe what this fork adds and how to use it day-to-day.

## What this fork adds

- **Datasets**: NEREL, NEREL-BIO, RuTermEval (Russian), GENIA, SecCol (cybersecurity events), CASIE, ACE2004/2005, CoNLL2003, BIONNE (biomedical, EN/RU).
- **Entity-type prompt strategies**: keyword (`_kw`), lexical-context (`_lao`), most-frequent-context (`_mfc`), contextual definitions (`_con`), structural/nested (`_struct-nested`). Defined in JSONL files at the repo root (see `entity_types_*.json`, `et_*.json`).
- **Inference entry points**:
  - `text2ner.py` — free-text input (Russian or English) → NER predictions.
  - `inference.py` — batched inference driven by a JSON config.
- **Cross-validation** runner (`run_cross_val.py`).
- **Memory / parameter diagnostics** (`memory_profiler_training.py`, `params_count.py`).
- **BRAT ↔ HuggingFace Dataset converters** in `data_preproc/`.

## Repository layout

```
.
├── run_ner.py             # main training entry point (HuggingFace Trainer + Binder)
├── run_cross_val.py       # k-fold cross-validation runner
├── train_binder.py        # alternative trainer encapsulating training as a class
├── text2ner.py            # free-text → NER inference (the script you usually want)
├── inference.py           # config-driven batch inference
├── memory_profiler_training.py
├── params_count.py
├── binder_metrics_to_csv.py
│
├── src/                   # canonical library code (model, trainer, utils) — see src/README.md
├── conf/                  # JSON configs for every dataset / experiment
├── data_preproc/          # BRAT ↔ HuggingFace converters and prompt builders
├── archive/               # older / unused scripts kept for reference (see archive/README.md)
│
├── entity_types_*.json    # entity-type prompts (keyword, lao, mfc, con, def, struct-nested, bionne)
├── et_*.json              # smaller dataset-specific prompt files (bionne, rutermeval)
│
├── results/               # gitignored — stale result JSONs from past runs live here
└── logs/                  # gitignored — training/inference logs
```

`data/`, `checkpoints/`, `logs/`, `results/`, `wandb/`, `myvenv*/` are all gitignored.

## Installation

```bash
conda create -n binder -y python=3.9
conda activate binder
conda install pytorch==1.13 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers datasets wandb seqeval numpy nltk safetensors psutil
```

(See `CLAUDE.md` for newer versions some scripts have been tested against.)

## Training

All training goes through `run_ner.py` (or `train_binder.py` for the class-based variant) with a single JSON config:

```bash
python run_ner.py conf/<dataset-config>.json
```

Examples:

```bash
# Russian NEREL with the keyword prompt
python run_ner.py conf/nerel-bio-new.json

# BIONNE (biomedical, English)
python run_ner.py conf/bionne/en/baseline/33.json

# GENIA flat NER
python run_ner.py conf/genia/flat.json

# ACE2005 (original BINDER setup)
python run_ner.py conf/ace05.json
```

Shell wrappers in the repo root (`run.sh`, `train.sh`, `rutermeval.sh`, `run_seqlen_experiment.sh`, …) chain multiple `run_ner.py` invocations and clear the HF dataset cache between runs. They are project-specific scripts kept for reproducibility — read the file before running.

### Cross-validation

```bash
python run_cross_val.py conf/seccol/cross-val.json
```

## Inference

### Free-text input → NER predictions

```bash
python text2ner.py
```

`text2ner.py` reads its settings from `conf/inference/text2ner/inference-config.json` (or `conf/inference/text2ner-nerel/inference-config.json`). The default points to the published `fulstock/NEREL-binder` checkpoint on HuggingFace. The script sentence-tokenizes input with NLTK (Russian punkt) and word-tokenizes with a robust `SafeWordTokenizer` (NLTK with regex fallback).

A reusable `BinderInference` class lives in `src/inference.py` if you want to call inference programmatically.

### Batched inference driven by a config

```bash
python inference.py conf/inference/<config>.json
```

Use this when you have a HuggingFace-format dataset and want metrics + predictions written to disk.

## Data preparation

```bash
# BRAT-format annotations → HuggingFace Dataset JSON (NEREL-style datasets)
python data_preproc/brat_to_hfds.py \
    --brat_dataset_path ./data/NEREL \
    --tags_path ./data_preproc/nerel.tags \
    --hfds_output_path ./data/NEREL-binder

# Predictions back to BRAT format
python data_preproc/hfds_to_brat.py

# ACE / CoNLL conversion
python data_preproc/convert_to_hf_ds_format.py <input.json> <output.json>
```

More details in `data_preproc/README.md`.

## Configuration: key fields

| Field | Purpose |
|---|---|
| `model_name_or_path` | Base pretrained model (e.g. `DeepPavlov/rubert-base-cased`, `FacebookAI/xlm-roberta-base`). |
| `binder_model_name_or_path` | Trained Binder checkpoint, for inference or continued training. |
| `entity_type_file` | JSONL prompt file at the repo root (e.g. `entity_types_kw.json`). |
| `dataset_entity_types` | Array of entity-type names to use from that file. |
| `dataset_name` | Filters entity types from `entity_type_file` by `dataset` field. |
| `max_seq_length` | Typically 192. |
| `doc_stride` | Typically 16. |
| `use_span_width_embedding` | Usually `true`. |
| `weighted_loss` | Optional class-frequency-weighted loss. |
| `do_neutral_spans` | Enable neutral-span handling. |

`CLAUDE.md` has a longer architecture overview.

## Additional documentation

- `src/README.md` — what each module in `src/` does.
- `data_preproc/README.md` — details for ACE/CoNLL preprocessing.
- `archive/README.md` — what the scripts in `archive/` are and why they're not on the main path.
- `CLAUDE.md` — full architectural overview and config field reference.

---

## Original paper

This is a fork of [BINDER (Microsoft Research)](https://github.com/microsoft/binder).

> Sheng Zhang, Hao Cheng, Jianfeng Gao, Hoifung Poon.
> *Optimizing Bi-Encoder for Named Entity Recognition via Contrastive Learning.* ICLR 2023.
> [openreview](https://openreview.net/forum?id=9EAQVEINuum) · [arXiv:2208.14565](https://arxiv.org/abs/2208.14565)

```bib
@article{zhang-etal-2022-binder,
  title={Optimizing Bi-Encoder for Named Entity Recognition via Contrastive Learning},
  author={Zhang, Sheng and Cheng, Hao and Gao, Jianfeng and Poon, Hoifung},
  journal={arXiv preprint arXiv:2208.14565},
  year={2022}
}
```

The original Microsoft licence (`LICENSE`), code of conduct (`CODE_OF_CONDUCT.md`), security (`SECURITY.md`), and support (`SUPPORT.md`) files are preserved as-is.
