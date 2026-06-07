# archive/

Scripts kept for reference / back-compatibility but no longer wired into any shell script or imported by any active module. Nothing in the live codebase depends on these files.

If you discover that one of these is still needed, move it back to the repo root rather than referencing it from this directory.

| File | What it is | Replaced by |
|---|---|---|
| `trainer.py` | Older copy of the `BinderTrainer` + data collator. | `src/trainer.py` (canonical; richer and actually imported by entry points) |
| `utils.py` | Older copy of post-processing / metrics utilities. | `src/utils.py` (canonical) |
| `binder_proc.py` | Early variant of the training entry point. Same docstring as `run_ner.py`, predates it. | `run_ner.py` |
| `text2ner_original.py` | First version of the free-text→NER inference script. | `text2ner.py` |
| `text2ner_optimized.py` | Intermediate refactor of `text2ner_original.py` on the way to the current `text2ner.py`. | `text2ner.py` |
| `convert_unicode.py` | Seven-line one-off that re-encodes `predict_predictions.json` to UTF-8. Hardcodes a Windows path (`S:/HRCode/...`). | — (run ad hoc if ever needed) |
