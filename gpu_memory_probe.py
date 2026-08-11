#!/usr/bin/env python3
"""
GPU Memory Profiler for Binder SecCol model inference.
Measures peak GPU memory usage during forward pass.

Purpose: confirm whether Binder fits on Tesla P100 (16 GB).

Usage:
    cd /path/to/binder/repo
    python gpu_memory_probe.py --checkpoint ./checkpoints/seccol_NEREL-attack-binder
    python gpu_memory_probe.py --checkpoint ./checkpoints/seccol_NEREL-attack-binder --batch-size 8
"""

import argparse
import gc
import os
import sys
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import torch
from transformers import AutoTokenizer
from datasets import load_dataset, disable_progress_bar

disable_progress_bar()

from src.model import Binder

SECCOL_DATASET = "seccol_events_texts_1500_new"
SECCOL_TYPES = [
    "ATTACK", "DAMAGE", "DEVICE", "DISCOVER_VULNERABILITY", "FILE",
    "GPE", "HACKER", "HACKER_GROUP", "INFOSOURCE", "MALWARE",
    "MONEY", "ORGANIZATION", "PATCH_VULNERABILITY", "PERSON",
    "PROGRAM_SYSTEM", "PROTECTION", "SPECIALIST", "TIME",
    "VULNERABILITY", "WEBSITE",
]
DEFAULT_ENTITY_FILE = "entity_types_kw.json"


def sep(title=""):
    if title:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
    else:
        print(f"{'='*60}")


def count_params(model):
    text_enc = sum(p.numel() for p in model.text_encoder.parameters())
    type_enc = sum(p.numel() for p in model.type_encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    other = total - text_enc - type_enc
    return {"text_encoder": text_enc, "type_encoder": type_enc, "other": other, "total": total}


def gb_str(val_bytes):
    return f"{val_bytes / (1024**3):.2f} GB"


def load_entity_types(tokenizer, entity_type_file, max_seq_length):
    raw = load_dataset("json", data_files=entity_type_file)["train"]
    raw = raw.filter(
        lambda ex: ex["dataset"] == SECCOL_DATASET and ex["name"] in SECCOL_TYPES
    )
    raw = raw.sort("name")
    id_to_str = [et["name"] for et in raw]

    tokenized = raw.map(
        lambda examples: tokenizer(
            examples["description"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        ),
        batched=True,
        remove_columns=raw.column_names,
    )

    input_ids = torch.tensor(tokenized["input_ids"])
    attention_mask = torch.tensor(tokenized["attention_mask"])
    token_type_ids = (
        torch.tensor(tokenized["token_type_ids"])
        if "token_type_ids" in tokenized
        else None
    )
    return input_ids, attention_mask, token_type_ids, id_to_str


def tokenize_text(tokenizer, text, max_seq_length):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")


def run_forward(model, input_ids, attention_mask, token_type_ids,
                type_input_ids, type_attention_mask, type_token_type_ids):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            type_input_ids=type_input_ids,
            type_attention_mask=type_attention_mask,
            type_token_type_ids=type_token_type_ids,
        )

    peak = torch.cuda.max_memory_allocated()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    return {"peak_allocated": peak, "current_allocated": allocated, "reserved": reserved}


def build_russian_text(target_tokens_approx):
    filler = (
        "Специалисты по кибербезопасности обнаружили новую уязвимость "
        "в программном обеспечении компании Microsoft Windows Server. "
        "Хакерская группировка использовала вредоносное программное обеспечение "
        "для атаки на корпоративные сети и получения доступа к конфиденциальным данным. "
        "Организация выпустила патч безопасности для устранения данной уязвимости. "
        "Эксперты рекомендуют установить обновление для защиты от подобных атак."
    )
    # ruBERT tokenizes ~1.5 chars per token on average, so ~1.5 * target
    return filler * max(1, target_tokens_approx * 3 // len(filler) + 1)


def main():
    parser = argparse.ArgumentParser(description="Binder GPU memory profiler")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Binder checkpoint directory")
    parser.add_argument("--entity-types", type=str, default=DEFAULT_ENTITY_FILE,
                        help="Path to entity types JSONL file")
    parser.add_argument("--max-seq-length", type=int, default=192,
                        help="Max sequence length in tokens")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for testing")
    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt):
        print(f"ERROR: Checkpoint directory not found: {ckpt}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_gpu_mem / (1024**3)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    sep("1. LOADING MODEL")
    print(f"  Checkpoint : {ckpt}")
    print(f"  GPU        : {device_name} ({total_gb:.1f} GB)")

    model = Binder.from_pretrained(ckpt)
    params = count_params(model)

    print(f"\n  Parameter breakdown:")
    print(f"    text_encoder : {params['text_encoder']:>15,}  ({params['text_encoder'] * 4 / 1024**2:>7.0f} MB FP32,  {params['text_encoder'] * 2 / 1024**2:>7.0f} MB FP16)")
    print(f"    type_encoder : {params['type_encoder']:>15,}  ({params['type_encoder'] * 4 / 1024**2:>7.0f} MB FP32,  {params['type_encoder'] * 2 / 1024**2:>7.0f} MB FP16)")
    print(f"    other        : {params['other']:>15,}  ({params['other'] * 4 / 1024**2:>7.0f} MB FP32,  {params['other'] * 2 / 1024**2:>7.0f} MB FP16)")
    print(f"    TOTAL        : {params['total']:>15,}  ({params['total'] * 4 / 1024**2:>7.0f} MB FP32,  {params['total'] * 2 / 1024**2:>7.0f} MB FP16)")

    weight_mb_fp32 = params['total'] * 4 / 1024**2
    weight_mb_fp16 = params['total'] * 2 / 1024**2

    # ------------------------------------------------------------------
    # 2. Load tokenizer and entity types
    # ------------------------------------------------------------------
    sep("2. LOADING TOKENIZER & ENTITY TYPES")

    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
    except Exception:
        print("  Tokenizer not found in checkpoint, falling back to DeepPavlov/rubert-base-cased")
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

    type_input_ids, type_attention_mask, type_token_type_ids, id_to_str = \
        load_entity_types(tokenizer, args.entity_types, args.max_seq_length)

    num_types = len(id_to_str)
    type_seq_len = type_input_ids.shape[1]
    print(f"  Entity types : {num_types} ({', '.join(id_to_str)})")
    print(f"  Type seq len : {type_seq_len} tokens")

    # ------------------------------------------------------------------
    # 3. Prepare test texts
    # ------------------------------------------------------------------
    sep("3. PREPARING TEST INPUTS")

    short_text  = "Компания Microsoft выпустила обновление безопасности."
    medium_text = build_russian_text(50)
    long_text   = build_russian_text(180)

    test_texts = [
        ("short  (~10 tokens)", short_text),
        ("medium (~50 tokens)", medium_text),
        ("long  (~180 tokens)", long_text),
    ]

    # Print actual token counts
    print()
    for label, text in test_texts:
        ids, _, _ = tokenize_text(tokenizer, text, args.max_seq_length)
        real_tokens = (ids > 0).sum().item() - 2  # subtract CLS and SEP
        print(f"  {label:<25} -> {real_tokens} real text tokens ({len(text)} chars)")

    # ------------------------------------------------------------------
    # 4. Memory profiling
    # ------------------------------------------------------------------
    sep("4. MEMORY PROFILING (peak GPU allocated during forward pass)")

    results = []

    for dtype_name, model_convert in [
        ("FP32", lambda m: m.float()),
        ("FP16", lambda m: m.half()),
    ]:
        model_convert(model)
        model.eval()

        ti_cuda = type_input_ids.cuda()
        ta_cuda = type_attention_mask.cuda()
        tt_cuda = type_token_type_ids.cuda() if type_token_type_ids is not None else None

        ti_batch = ti_cuda.unsqueeze(0).repeat(args.batch_size, 1, 1)
        ta_batch = ta_cuda.unsqueeze(0).repeat(args.batch_size, 1, 1)
        tt_batch = tt_cuda.unsqueeze(0).repeat(args.batch_size, 1, 1) if tt_cuda is not None else None

        print(f"\n  --- {dtype_name}, batch_size={args.batch_size} ---")

        for label, text in test_texts:
            input_ids, attention_mask, token_type_ids = tokenize_text(
                tokenizer, text, args.max_seq_length
            )
            input_ids = input_ids.repeat(args.batch_size, 1).cuda()
            attention_mask = attention_mask.repeat(args.batch_size, 1).cuda()
            if token_type_ids is not None:
                token_type_ids = token_type_ids.repeat(args.batch_size, 1).cuda()

            # Warmup (not counted)
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                type_input_ids=ti_batch,
                type_attention_mask=ta_batch,
                type_token_type_ids=tt_batch,
            )

            # Measured run
            mem = run_forward(
                model, input_ids, attention_mask, token_type_ids,
                ti_batch, ta_batch, tt_batch,
            )

            token_count = attention_mask.sum().item()
            results.append({
                "dtype": dtype_name,
                "label": label,
                "tokens": token_count,
                "peak": mem["peak_allocated"],
                "alloc": mem["current_allocated"],
                "reserved": mem["reserved"],
            })

            print(f"    {label:<22} tokens={token_count:>4}  "
                  f"peak={gb_str(mem['peak_allocated']):>7}  "
                  f"resident={gb_str(mem['current_allocated'])}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    sep("5. SUMMARY")

    max_peak_fp32 = max(r["peak"] for r in results if r["dtype"] == "FP32")
    max_peak_fp16 = max(r["peak"] for r in results if r["dtype"] == "FP16")
    max_peak = max(max_peak_fp32, max_peak_fp16)

    print(f"\n  {'Metric':<35} {'Value':>12}")
    print(f"  {'-'*48}")
    print(f"  {'GPU total memory':<35} {total_gb:>9.1f} GB")
    print(f"  {'Model weights (FP32)':<35} {weight_mb_fp32:>9.0f} MB")
    print(f"  {'Model weights (FP16)':<35} {weight_mb_fp16:>9.0f} MB")
    print(f"  {'Peak inference memory (FP32)':<35} {max_peak_fp32/(1024**3):>9.2f} GB")
    print(f"  {'Peak inference memory (FP16)':<35} {max_peak_fp16/(1024**3):>9.2f} GB")

    fits_fp32_16 = max_peak_fp32 < 13 * 1024**3
    fits_fp16_16 = max_peak_fp16 < 13 * 1024**3
    fp32_pct = max_peak_fp32 / (16 * 1024**3) * 100
    fp16_pct = max_peak_fp16 / (16 * 1024**3) * 100

    print(f"\n  {'='*55}")
    print(f"  P100 (16 GB) COMPATIBILITY")
    print(f"  {'='*55}")
    print(f"  FP32 inference: peak {max_peak_fp32/(1024**3):.1f} GB ({fp32_pct:.0f}% of 16 GB)  -> {'FITS' if fits_fp32_16 else 'DOES NOT FIT'}")
    print(f"  FP16 inference: peak {max_peak_fp16/(1024**3):.1f} GB ({fp16_pct:.0f}% of 16 GB)  -> {'FITS' if fits_fp16_16 else 'DOES NOT FIT'}")

    if fits_fp16_16:
        headroom = 16 - max_peak_fp16 / (1024**3)
        print(f"\n  FP16 mode leaves {headroom:.1f} GB headroom on P100.")
        print(f"  Recommended: run with --fp16 flag, batch_size=1.")
    elif fits_fp32_16:
        headroom = 16 - max_peak_fp32 / (1024**3)
        print(f"\n  FP32 mode fits with {headroom:.1f} GB headroom on P100.")
        print(f"  FP16 mode gives even more room.")
    else:
        print(f"\n  WARNING: Peak exceeds 16 GB. Reduce max_seq_length or use smaller model.")

    # Per-run detail table
    print(f"\n  Detailed per-run results:")
    print(f"  {'Dtype':<6} {'Input':<22} {'Tokens':>7} {'Peak':>8} {'Resident':>8}  {'Fits P100?':>12}")
    print(f"  {'-'*72}")
    for r in results:
        fits = "YES" if r["peak"] < 13 * 1024**3 else "NO"
        print(f"  {r['dtype']:<6} {r['label']:<22} {r['tokens']:>7} {gb_str(r['peak']):>8} {gb_str(r['alloc']):>8}  {fits:>12}")
    print()


if __name__ == "__main__":
    main()
