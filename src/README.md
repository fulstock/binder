# `src/` — Binder library code

Canonical implementation of the Binder bi-encoder model and its training / inference plumbing. All top-level entry-point scripts (`run_ner.py`, `text2ner.py`, `inference.py`, `run_cross_val.py`, `train_binder.py`) import from this package.

> If you find sibling copies of `trainer.py` / `utils.py` in `archive/`, those are stale — only the versions in `src/` are imported by live code.

## Modules

### `config.py`
- **`BinderConfig`** — `transformers.PretrainedConfig` subclass.
  Loss weights (`start_loss_weight`, `end_loss_weight`, `span_loss_weight`, `threshold_loss_weight`, `ner_loss_weight`), `linear_size`, `init_temperature`, optional `use_span_width_embedding`.

### `model.py`
- **`Binder`** — `transformers.PreTrainedModel` subclass. Two BERT encoders (`text_encoder`, `type_encoder`) initialised from the same pretrained checkpoint, projecting text spans and entity-type descriptions into a shared vector space. Produces `start_scores`, `end_scores`, `span_scores`.
- **`BinderModelOutput`** — dataclass for the forward-pass output.
- Helpers: `masked_log_softmax`, `contrastive_loss`, `tiny_value_of_dtype`.

### `trainer.py`
- **`BinderTrainer`** — `transformers.Trainer` subclass with Binder-specific loss, evaluation, and prediction.
- **`BinderDataCollator`** — dynamic padding and entity-type tensor injection.
- **`Span`** — dataclass used by the collator.

### `utils.py`
Post-processing and metrics:
- **`Annotation`** — dataclass: `(id, entity_type, start_char, end_char, text)`. The standard entity representation across the pipeline.
- **`compute_tp_fn_fp`**, **`compute_precision_recall_f1`**, **`compute_everything`** — metrics primitives.
- **`postprocess_nested_predictions`** / **`postprocess_nested_predictions_with_threshold`** — convert raw model scores into entity annotations.
- **`remove_overlaps`** — non-maximum suppression over span predictions.
- **`convert_to_iob`** — span annotations → IOB tagging (for compatibility with `seqeval`).
- **`error_analysis`** — per-type confusion breakdown.

### `inference.py`
Reusable inference machinery:
- **`BinderInference`** — high-level class for loading a checkpoint and running predictions on arbitrary input.
- **`SafeWordTokenizer`** — NLTK word tokenizer with a regex fallback for malformed input.
- **`ModelArguments` / `DataTrainingArguments`** — `HfArgumentParser` dataclasses (mirror those in `run_ner.py`).
- **`evaluate_on_dataset`**, **`compute_metrics`**, **`print_metrics`**, **`load_binder_dataset`** — batch-evaluation helpers.
- **`TimingStats`** — latency tracking.

### `memory_callback.py`
- **`MemoryUsageCallback`** — `TrainerCallback` that logs GPU memory at epoch end and at each evaluation step.

## How a training run flows through `src/`

1. `run_ner.py` parses its JSON config with `HfArgumentParser` into `ModelArguments`, `DataTrainingArguments`, `TrainingArguments`.
2. It builds a `BinderConfig` and instantiates `Binder(config)` (both encoders init from the same pretrained checkpoint).
3. It loads the dataset, builds entity-type tensors from the `entity_type_file` prompt JSONL, and creates a `BinderDataCollator`.
4. A `BinderTrainer` runs training; `MemoryUsageCallback` logs memory; `src/utils.py` post-processes predictions into `Annotation` sets and computes F1 via `compute_tp_fn_fp` + `compute_precision_recall_f1`.

The inference path (`text2ner.py`, `inference.py`) re-uses steps 1-3 with `do_train=false` and then calls into `BinderInference` (or the trainer's `predict`) plus the same post-processing utilities.
