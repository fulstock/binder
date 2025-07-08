#!/usr/bin/env python3
"""
Three-way performance comparison with batch awareness.

Compares:
1. Text2NER (Optimized BinderInference) – single & batch
2. Binder_Proc (Original BinderInference) – single only
3. BERT NER baseline – single only

The script reuses the concise, human-readable reporting style from `three_way_comparison.py`
while adding the batch processing support demonstrated in `batch_aware_comparison.py`.
"""

import os
import sys
import time
import gc
import psutil
import torch
import warnings
from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
#  Implementations
# -----------------------------------------------------------------------------

try:
    from text2ner import BinderInference as TextNerInference
    from binder_proc import BinderInference as BinderProcInference
    print("✅ Successfully imported Binder modules")
except ImportError as e:
    print(f"❌ Failed to import Binder modules: {e}")
    sys.exit(1)


class BertNerBaseline:
    """Minimal BERT NER baseline for speed reference"""

    def __init__(self, model_name: str = "DeepPavlov/rubert-base-cased", device: str = "cpu"):
        self.device = device
        self.model_name = model_name

        print(f"Loading BERT NER baseline: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        try:
            ner_model_name = "DeepPavlov/rubert-base-cased-conversational"
            self.model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
            print(f"Loaded conversational model: {ner_model_name}")
        except Exception:
            from transformers import BertConfig, BertForTokenClassification

            config = BertConfig.from_pretrained(model_name)
            config.num_labels = 21  # Generic label space
            self.model = BertForTokenClassification.from_pretrained(
                model_name, config=config, ignore_mismatched_sizes=True
            )
            print("Created generic token classification head")

        self.model.to(device)
        self.model.eval()
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            aggregation_strategy="simple",
        )

    def predict(self, text: str):  # type: ignore[override]
        try:
            entities = self.ner_pipeline(text)
            results = []
            for ent in entities:
                if ent.get("score", 0) < 0.1:
                    continue
                start, end = ent.get("start", 0), ent.get("end", 0)
                label = ent.get("entity_group", ent.get("entity", "MISC"))
                results.append((start, end, label, text[start:end]))
            return results
        except Exception as e:
            print(f"BERT prediction error: {e}")
            return []


# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

def get_memory_usage() -> float:
    """Return current RSS memory usage (MB)."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds/60:.1f}m {seconds%60:.1f}s"


def get_test_files(max_files: int = 8) -> List[str]:
    test_dir = "./data/seccol_events_texts_1500_new2-div/test"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return []

    txt_files: List[str] = []
    for fn in sorted(os.listdir(test_dir)):
        if fn.endswith(".txt") and len(txt_files) < max_files:
            txt_files.append(os.path.join(test_dir, fn))
    return txt_files


# -----------------------------------------------------------------------------
#  Core benchmarking routine
# -----------------------------------------------------------------------------

def benchmark_impl(
    impl_cls,
    cfg_or_params,
    test_files: List[str],
    name: str,
    use_batch: bool = False,
):
    """Benchmark a single implementation in either single or batch mode.

    Args:
        impl_cls: class or callable to create inference object.
        cfg_or_params: path to config (Binder) or kwargs dict (BERT baseline).
        test_files: list of file paths.
        name: identifier for logs.
        use_batch: if True and implementation supports `predict_batch`, run in batch mode.
    Returns:
        Dict with metrics or None on failure.
    """
    mode = "batch" if use_batch else "single"
    print("\n" + "=" * 70)
    print(f"Testing {name} – {mode.upper()} mode")
    print("=" * 70)

    try:
        gc.collect()
        start_mem = get_memory_usage()
        print(f"Initial memory: {start_mem:.1f}MB")

        # Initialise implementation
        t0 = time.time()
        if name.startswith("BERT"):
            inf = impl_cls(**(cfg_or_params or {}))
        else:
            if isinstance(cfg_or_params, str) and not os.path.exists(cfg_or_params):
                print(f"❌ Config not found: {cfg_or_params}")
                return None
            inf = impl_cls(config_path=cfg_or_params, device="cpu")
        init_time = time.time() - t0
        after_init_mem = get_memory_usage()

        print(f"✅ Init done in {format_time(init_time)}; RSS {after_init_mem:.1f}MB")

        # Warm-up
        try:
            _ = inf.predict("Это тестовый текст для прогрева модели.")
            print("🔥 Warm-up completed")
        except Exception as e:
            print(f"⚠️ Warm-up failed: {e}")

        # Read texts
        texts: List[str] = []
        for fp in test_files:
            try:
                txt = open(fp, "r", encoding="utf-8").read().strip()
                if txt and len(txt) <= 10000:
                    texts.append(txt)
            except Exception as e:
                print(f"⚠️ Could not read {os.path.basename(fp)}: {e}")
        if not texts:
            print("❌ No valid texts for processing")
            return None

        # Batch-specific prep
        if use_batch and hasattr(inf, "enable_batch_mode"):
            try:
                inf.enable_batch_mode()
            except Exception:
                pass  # Ignore silently if method errors

        t_process_start = time.time()

        if use_batch and hasattr(inf, "predict_batch"):
            # Conservative batch sizing handled inside predict_batch / enable_batch_mode
            predictions_lists = inf.predict_batch(texts)
        else:
            predictions_lists = []
            for i, t_ in enumerate(texts, 1):
                preds = inf.predict(t_)
                predictions_lists.append(preds)
                if i % 3 == 0:
                    print(f"Progress: {i}/{len(texts)}")

        proc_time = time.time() - t_process_start
        peak_mem = max(after_init_mem, get_memory_usage())

        processed_files = len(predictions_lists)
        total_entities = sum(len(p) for p in predictions_lists)
        total_chars = sum(len(t) for t in texts)

        files_per_sec = processed_files / proc_time if proc_time > 0 else 0.0
        chars_per_sec = total_chars / proc_time if proc_time > 0 else 0.0
        entities_per_file = total_entities / processed_files if processed_files else 0.0

        print(f"\n{name} – {mode} results:")
        print(f"⏱  Processing: {format_time(proc_time)} (throughput {files_per_sec:.2f} files/s)")
        print(f"📝 Chars/s: {chars_per_sec:.0f}; Entities/file: {entities_per_file:.1f}")
        print(f"💾 Peak RSS: {peak_mem:.1f}MB")

        return {
            "name": name,
            "mode": mode,
            "init_time": init_time,
            "processing_time": proc_time,
            "files_per_second": files_per_sec,
            "chars_per_second": chars_per_sec,
            "entities_per_file": entities_per_file,
            "peak_memory": peak_mem,
        }

    except Exception as e:
        print(f"❌ Error benchmarking {name}: {e}")
        return None


# -----------------------------------------------------------------------------
#  Reporting helpers
# -----------------------------------------------------------------------------

def print_results_table(results: List[Dict]):
    if not results:
        print("No results to display")
        return

    header = f"{'Implementation':<25} {'Mode':<7} {'Init':<8} {'Proc':<8} {'Files/s':<8} {'Chars/s':<10} {'Ent/file':<8} {'Mem(MB)':<8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        files_per_sec = f"{r['files_per_second']:.2f}"
        chars_per_sec = f"{r['chars_per_second']:.0f}"
        ent_per_file = f"{r['entities_per_file']:.1f}"
        peak_mem = f"{r['peak_memory']:.1f}"
        print(
            f"{r['name']:<25} {r['mode']:<7} {format_time(r['init_time']):<8} {format_time(r['processing_time']):<8} "
            f"{files_per_sec:<8} {chars_per_sec:<10} {ent_per_file:<8} {peak_mem:<8}"
        )

    # Speed ranking
    print("\nSpeed ranking (by files/s):")
    ranking = sorted(results, key=lambda x: x["files_per_second"], reverse=True)
    for idx, r in enumerate(ranking, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"{idx}.")
        print(f"  {medal} {r['name']} ({r['mode']}) – {r['files_per_second']:.2f} files/s")

    # Highlight speedup of optimized over original
    orig = next((r for r in results if r["name"].startswith("Binder_Proc")), None)
    opt_batch = next(
        (r for r in results if r["name"].startswith("Text2NER") and r["mode"] == "batch"),
        None,
    )
    if orig and opt_batch and orig["files_per_second"] > 0:
        speedup = opt_batch["files_per_second"] / orig["files_per_second"]
        print(f"\n🚀 Text2NER batch is {speedup:.2f}× faster than Binder_Proc single")


# -----------------------------------------------------------------------------
#  Main entry
# -----------------------------------------------------------------------------

def main():
    print("🚀 Three-way batch-aware comparison")
    test_files = get_test_files(max_files=8)
    if not test_files:
        return

    results: List[Dict] = []

    # Binder_Proc (Original) – single
    res_binder = benchmark_impl(
        BinderProcInference,
        "./inference/inference-config.json",
        test_files,
        "Binder_Proc (Original)",
        use_batch=False,
    )
    if res_binder:
        results.append(res_binder)

    # Text2NER (Optimized) – single
    res_text_single = benchmark_impl(
        TextNerInference,
        "./inference/inference-config-optimized.json",
        test_files,
        "Text2NER (Optimized)",
        use_batch=False,
    )
    if res_text_single:
        results.append(res_text_single)

    # Text2NER (Optimized) – batch
    res_text_batch = benchmark_impl(
        TextNerInference,
        "./inference/inference-config-optimized.json",
        test_files,
        "Text2NER (Optimized)",
        use_batch=True,
    )
    if res_text_batch:
        results.append(res_text_batch)

    # BERT baseline – single
    res_bert = benchmark_impl(
        BertNerBaseline,
        {},
        test_files,
        "BERT NER Baseline",
        use_batch=False,
    )
    if res_bert:
        results.append(res_bert)

    print_results_table(results)

    # System info
    print("\nSystem info:")
    mem = psutil.virtual_memory()
    print(f"Memory total/available: {mem.total / 1e9:.1f}GB / {mem.available / 1e9:.1f}GB")
    print(f"CPU cores: {psutil.cpu_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        print("GPU: not available")

    print("\n✅ Comparison complete")


if __name__ == "__main__":
    main() 