#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Binder for named entity recognition.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import re
import psutil
import gc
import time
import threading
import platform
import statistics
import traceback
import json
import csv

# Fix Windows console UTF-8 encoding for emoji support
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import datasets
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# Ensure we have the necessary imports for model loading
import safetensors

# Additional imports for robust model loading
try:
    import huggingface_hub
    HF_HUB_AVAILABLE = True
except ImportError:
    print("Warning: huggingface_hub not available. Some model loading strategies will be skipped.")
    HF_HUB_AVAILABLE = False

from transformers.trainer_utils import get_last_checkpoint

from src.config import BinderConfig
from src.model import Binder
from src.trainer import BinderDataCollator, BinderTrainer
from src import utils as postprocess_utils

import nltk
import torch

from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer

# Custom Word Tokenizer that actually implements span_tokenize
class SafeWordTokenizer:
    """A safe word tokenizer with proper span_tokenize implementation"""
    
    def __init__(self):
        # Word boundary pattern - matches sequences of word characters
        self.word_pattern = re.compile(r'\b\w+\b')
        # Try to use NLTK if span_tokenize works, otherwise use regex
        self._use_nltk = False
        try:
            from nltk.tokenize import NLTKWordTokenizer
            test_tokenizer = NLTKWordTokenizer()
            # Test if span_tokenize is implemented
            list(test_tokenizer.span_tokenize("test text"))
            self.nltk_tokenizer = test_tokenizer
            self._use_nltk = True
            print("Using NLTK word tokenizer")
        except (NotImplementedError, Exception) as e:
            print(f"NLTK span_tokenize not available ({e}). Using regex fallback.")
            
    def tokenize(self, text):
        """Tokenize text into words"""
        if self._use_nltk:
            return self.nltk_tokenizer.tokenize(text)
        else:
            return self.word_pattern.findall(text)
    
    def span_tokenize(self, text):
        """Return word spans (start, end) positions"""
        if self._use_nltk:
            try:
                # Ensure we return a generator that actually works
                spans = list(self.nltk_tokenizer.span_tokenize(text))
                if spans:  # If NLTK actually found spans
                    for span in spans:
                        yield span
                else:
                    # Fallback to regex if NLTK returns empty
                    for match in self.word_pattern.finditer(text):
                        yield (match.start(), match.end())
            except Exception as e:
                print(f"NLTK span_tokenize failed during execution: {e}. Using regex fallback.")
                # Fallback to regex
                for match in self.word_pattern.finditer(text):
                    yield (match.start(), match.end())
        else:
            for match in self.word_pattern.finditer(text):
                yield (match.start(), match.end())

@dataclass
class ModelArguments:
    """
    Arguments for Binder.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    binder_model_name_or_path : str = field(
        metadata={"help" : "Path to pretrained Binder model from huggingface.co/models or local path to a saved Binder module."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    hidden_dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout rate for hidden states."}
    )
    use_span_width_embedding: bool = field(
        default=False, metadata={"help": "Use span width embeddings."}
    )
    linear_size: int = field(
        default=128, metadata={"help": "Size of the last linear layer."}
    )
    init_temperature: float = field(
        default=0.07, metadata={"help": "Init value of temperature used in contrastive loss."}
    )
    start_loss_weight: float = field(
        default=0.2, metadata={"help": "NER span start loss weight."}
    )
    end_loss_weight: float = field(
        default=0.2, metadata={"help": "NER span end loss weight."}
    )
    span_loss_weight: float = field(
        default=0.6, metadata={"help": "NER span loss weight."}
    )
    threshold_loss_weight: float = field(
        default=0.5, metadata={"help": "NER threshold loss weight."}
    )
    ner_loss_weight: float = field(
        default=0.5, metadata={"help": "NER loss weight."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use, from which it will decide entity types to use."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_span_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an entity span."
        },
    )
    do_neutral_spans: bool = field(
        default=False, metadata={"help": "Produce neutral spans in flat2nested setting."}
    )
    predict_entities_id_file: str = field(
        default = "", metadata = {"help": "Prediction ids for logits."}
    )
    predict_logits_input_file: str = field(
        default = "", metadata = {"help": "Prediction logits for mask."}
    )
    entity_type_file: str = field(
        default=None,
        metadata={"help": "The entity type file contains all entity type names, descriptions, etc."},
    )
    dataset_entity_types: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "The entity types of this dataset, which are only a part of types in the entity type file."},
    )
    entity_type_key_field: Optional[str] = field(
        default="name",
        metadata={"help": "The field in the entity type file that will be used as key to sort entity types."},
    )
    entity_type_desc_field: Optional[str] = field(
        default="description",
        metadata={"help": "The field in the entity type file that corresponds to entity descriptions."},
    )
    prediction_postprocess_func: Optional[str] = field(
        default="postprocess_nested_predictions",
        metadata={"help": "The name of prediction postprocess function."},
    )
    neutral_relative_threshold: Optional[float] = field(
        default=0.5,
        metadata={"help": "Relative threshold for neutral spans prediction."},
    )
    # wandb_project: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "The name of WANDB project."},
    # )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension == "json", "`train_file` should be a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension == "json", "`test_file` should be a json file."

# Import consolidated BinderInference from src.inference
from src.inference import BinderInference

from tqdm.auto import tqdm

datasets.disable_progress_bar()
datasets.logging.set_verbosity(datasets.logging.CRITICAL)
tags = set()

def read_file_with_fallback(file_path, primary_encoding='utf-8', fallback_encodings=None):
    """
    Read a file with automatic encoding fallback.

    Args:
        file_path: Path to file to read
        primary_encoding: Primary encoding to try (default: utf-8)
        fallback_encodings: List of fallback encodings to try (default: ['cp1251', 'windows-1251', 'latin-1'])

    Returns:
        str: File contents
    """
    if fallback_encodings is None:
        fallback_encodings = ['cp1251', 'windows-1251', 'latin-1']

    # Try primary encoding first
    encodings_to_try = [primary_encoding] + fallback_encodings

    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue

    # If all encodings fail, read with errors='replace'
    with open(file_path, "r", encoding=primary_encoding, errors='replace') as f:
        return f.read()

def stream_batch_prediction_demo(input_encoding='utf-8', output_encoding='utf-8'):
    """
    Resource-constrained streaming batch prediction example with anti-freeze protection:
    - Strict memory and timeout limits to prevent system freezing
    - Conservative batch sizes
    - Comprehensive error handling and resource monitoring
    - Automatic process priority management
    - Processes all files without artificial limits
    - ENHANCED: Detailed timing measurements for model benchmarking
    - ENHANCED: Configurable input/output encodings for cp1251 and other encodings

    Args:
        input_encoding: Encoding to use when reading .txt and .ann files (default: utf-8)
        output_encoding: Encoding to use when writing .txt and .ann files (default: utf-8)
    """
    print("="*80)
    print("STREAMING BATCH PREDICTION DEMO WITH DETAILED TIMING")
    print("="*80)

    # Configuration
    # inference_config_path = "./inference/inference-config-optimized.json"
    # dataset_path = "S:/HRCode/data/NEREL1.1/test"
    # predicted_dataset_path = "S:/HRCode/data/NEREL1.1/test_predicted"
    # inference_config_path = "./inference/inference-config-optimized.json"
    # dataset_path = "S:/HRCode/data/NEREL1.1/train"
    # predicted_dataset_path = "S:/HRCode/data/NEREL1.1/train_predicted"
    # inference_config_path = "./inference/inference-config-optimized.json"
    # dataset_path = "S:/HRCode/data/examples_check"
    # predicted_dataset_path = "S:/HRCode/data/examples_check_predicted_normal"
    # inference_config_path = "./inference/rubert-tiny2.json"
    # dataset_path = "S:/HRCode/data/NEREL1.1/test"
    # predicted_dataset_path = "S:/HRCode/data/NEREL1.1/test_predicted_rubert-tiny2"
    inference_config_path = "./inference/rubert-tiny2.json"
    dataset_path = "S:/HRCode/data/NEREL1.1/train"
    predicted_dataset_path = "S:/HRCode/data/NEREL1.1/train_predicted_rubert-tiny2"
    # inference_config_path = "./inference/small_nerel_inf.json"  # Use the config for small_nerel model
    # dataset_path = "S:/HRCode/data/NEREL1.1/test"
    # predicted_dataset_path = "S:/HRCode/data/NEREL1.1/test_predicted_small"
    BATCH_SIZE = 1024

    print(f"Input encoding: {input_encoding}")
    print(f"Output encoding: {output_encoding}")

    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {predicted_dataset_path}")
    print(f"Batch size: {BATCH_SIZE}")

    # Timing tracking dictionary
    timing_stats = {
        'model_path': inference_config_path,
        'initialization_time': 0,
        'warmup_time': 0,
        'total_prediction_time': 0,
        'batch_times': [],
        'per_file_times': [],
        'total_files': 0,
        'total_chars': 0,
        'total_entities': 0,
    }

    # Initialize inference engine
    print(f"\nInitializing BinderInference...")
    init_start = time.time()

    try:
        inf = BinderInference(
            config_path=inference_config_path,
            device="auto"
        )
        print("BinderInference initialized successfully")
    except Exception as e:
        print(f"Initialization failed: {e}")
        traceback.print_exc()
        return {}

    timing_stats['initialization_time'] = time.time() - init_start

    # Get the entity types the model was trained on for filtering evaluation
    model_entity_types = set(inf.data_args.dataset_entity_types)
    print(f"\nModel trained on {len(model_entity_types)} entity types: {sorted(model_entity_types)}")

    # Monitor memory usage after initialization
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory after init: {memory_usage:.1f}MB")
    print(f"Initialization time: {timing_stats['initialization_time']:.3f}s")

    # Warm up the model
    print("\nWarming up model...")
    warmup_start = time.time()
    try:
        inf.warm_up("Это тестовый текст для российских кибербезопасности.")
        timing_stats['warmup_time'] = time.time() - warmup_start
        print(f"Model warmup complete in {timing_stats['warmup_time']:.3f}s")
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")
        traceback.print_exc()

    init_time = timing_stats['initialization_time'] + timing_stats['warmup_time']
    print(f"Total initialization: {init_time:.2f}s")
    
    # Collect all text files
    print(f"\nCollecting text files...")
    text_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for f in sorted(files):
            if f.endswith(".txt"):
                text_files.append(os.path.join(root, f))
    
    print(f"Found {len(text_files)} text files")
    
    if not text_files:
        print("No text files found!")
        return {}
    
    # Create output directory
    os.makedirs(predicted_dataset_path, exist_ok=True)
    
    # Batch processing
    print(f"\nStarting batch processing...")
    total_start_time = time.time()
    
    text2pred = {}
    total_entities = 0
    total_chars = 0
    processed_files = 0
    failed_files = 0
    
    for batch_start in tqdm(range(0, len(text_files), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(text_files))
        batch_files = text_files[batch_start:batch_end]
        
        # Load batch texts
        batch_texts = []
        batch_filenames = []
        batch_file_mapping = {}
        
        for file_path in batch_files:
            try:
                text = read_file_with_fallback(file_path, input_encoding).strip()
                # Skip empty files
                if text:
                    batch_texts.append(text)
                    filename = os.path.basename(file_path)
                    batch_filenames.append(filename)
                    batch_file_mapping[len(batch_texts) - 1] = {
                        'filename': filename,
                        'text': text,
                        'file_path': file_path
                    }
                    total_chars += len(text)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                traceback.print_exc()
                failed_files += 1
                continue
        
        if not batch_texts:
            continue
        
        # Batch prediction with detailed timing
        batch_start_time = time.time()
        try:
            batch_predictions = inf.predict_batch(batch_texts)
            batch_time = time.time() - batch_start_time

            # Store batch timing
            batch_chars = sum(len(batch_file_mapping[i]['text']) for i in range(len(batch_texts)))
            timing_stats['batch_times'].append({
                'batch_num': batch_start // BATCH_SIZE + 1,
                'num_files': len(batch_texts),
                'num_chars': batch_chars,
                'time': batch_time,
                'files_per_sec': len(batch_texts) / batch_time,
                'chars_per_sec': batch_chars / batch_time
            })

            # Process batch results
            for i, (filename, predictions) in enumerate(zip(batch_filenames, batch_predictions)):
                text = batch_file_mapping[i]['text']
                text2pred[filename] = {
                    'text': text,
                    'preds': predictions
                }
                total_entities += len(predictions)
                processed_files += 1

                # Store per-file timing (estimated from batch)
                estimated_file_time = batch_time / len(batch_texts)
                timing_stats['per_file_times'].append({
                    'filename': filename,
                    'chars': len(text),
                    'entities': len(predictions),
                    'time_estimate': estimated_file_time
                })

                # Removed per-file logging for cleaner output

            # Performance logging with memory monitoring
            chars_per_sec = batch_chars / batch_time
            files_per_sec = len(batch_texts) / batch_time
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Removed per-batch logging for cleaner output
            
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            traceback.print_exc()
            failed_files += len(batch_texts)
            # Fallback to individual processing
            print(f"  Attempting individual fallback for {len(batch_texts)} files...")
            for i, text in enumerate(batch_texts):
                try:
                    pred = inf.predict(text)
                    filename = batch_filenames[i]
                    text2pred[filename] = {'text': text, 'preds': pred}
                    total_entities += len(pred)
                    processed_files += 1
                except Exception as e2:
                    print(f" Individual prediction failed for {batch_filenames[i]}: {e2}")
                    traceback.print_exc()
                    failed_files += 1
    
    total_processing_time = time.time() - total_start_time

    # Update timing stats
    timing_stats['total_prediction_time'] = total_processing_time
    timing_stats['total_files'] = processed_files
    timing_stats['total_chars'] = total_chars
    timing_stats['total_entities'] = total_entities

    # Calculate statistics
    if timing_stats['batch_times']:
        batch_times_only = [b['time'] for b in timing_stats['batch_times']]
        timing_stats['avg_batch_time'] = statistics.mean(batch_times_only)
        timing_stats['median_batch_time'] = statistics.median(batch_times_only)
        timing_stats['min_batch_time'] = min(batch_times_only)
        timing_stats['max_batch_time'] = max(batch_times_only)

    if processed_files > 0:
        timing_stats['avg_time_per_file'] = total_processing_time / processed_files
        timing_stats['throughput_files_per_sec'] = processed_files / total_processing_time
        timing_stats['throughput_chars_per_sec'] = total_chars / total_processing_time
        timing_stats['avg_entities_per_file'] = total_entities / processed_files

    # DEBUG: Show sample predictions
    print(f"\n" + "="*80)
    print(f"DEBUG: SAMPLE PREDICTIONS")
    print("="*80)
    sample_count = 0
    for filename, data in list(text2pred.items())[:3]:  # Show first 3 files
        print(f"\nFile: {filename}")
        print(f"Text length: {len(data['text'])} chars")
        print(f"Text preview: {data['text'][:100]}...")
        print(f"Predictions count: {len(data['preds'])}")
        if data['preds']:
            print(f"Sample predictions (first 5):")
            for i, pred in enumerate(data['preds'][:5]):
                if len(pred) == 3:
                    s, e, t = pred
                    text_span = data['text'][s:e]
                elif len(pred) == 4:
                    s, e, t, text_span = pred
                else:
                    text_span = "unknown"
                print(f"  {i+1}. [{s}:{e}] {t} = '{text_span}'")
        else:
            print("  No predictions!")
        sample_count += 1

    print(f"\n" + "="*80)
    print(f"PROCESSING SUMMARY")
    print("="*80)
    print(f"\n📊 FILES:")
    print(f"  • Files found: {len(text_files)}")
    print(f"  • Files processed: {processed_files}")
    print(f"  • Files failed: {failed_files}")
    print(f"  • Success rate: {(processed_files/(processed_files+failed_files)*100):.1f}%" if (processed_files+failed_files) > 0 else "N/A")

    print(f"\n📈 DATA:")
    print(f"  • Total entities found: {total_entities}")
    print(f"  • Total characters: {total_chars:,}")
    print(f"  • Average entities per file: {total_entities/processed_files:.1f}" if processed_files > 0 else "N/A")

    print(f"\n⏱️  TIMING:")
    print(f"  • Initialization time: {timing_stats['initialization_time']:.3f}s")
    print(f"  • Warmup time: {timing_stats['warmup_time']:.3f}s")
    print(f"  • Total prediction time: {total_processing_time:.3f}s")
    print(f"  • Average time per file: {timing_stats.get('avg_time_per_file', 0):.4f}s")

    if timing_stats.get('batch_times'):
        print(f"\n⏱️  BATCH TIMING STATS:")
        print(f"  • Number of batches: {len(timing_stats['batch_times'])}")
        print(f"  • Average batch time: {timing_stats['avg_batch_time']:.3f}s")
        print(f"  • Median batch time: {timing_stats['median_batch_time']:.3f}s")
        print(f"  • Min batch time: {timing_stats['min_batch_time']:.3f}s")
        print(f"  • Max batch time: {timing_stats['max_batch_time']:.3f}s")

    print(f"\n🚀 THROUGHPUT:")
    if processed_files > 0:
        print(f"  • Files per second: {timing_stats['throughput_files_per_sec']:.2f}")
        print(f"  • Characters per second: {timing_stats['throughput_chars_per_sec']:.0f}")
        print(f"  • Entities per second: {(total_entities/total_processing_time):.2f}")
    else:
        print(f"  • No files processed")

    # Final memory report
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    available_mem = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"\n💾 MEMORY:")
    print(f"  • Final memory usage: {final_memory:.1f}MB")
    print(f"  • Available memory: {available_mem:.1f}GB")

    # Save timing report to file
    timing_report_path = os.path.join(predicted_dataset_path, "timing_report.json")
    try:
        with open(timing_report_path, 'w', encoding='utf-8', newline = "") as f:
            json.dump(timing_stats, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Timing report saved to: {timing_report_path}")
    except Exception as e:
        print(f"\n⚠️  Failed to save timing report: {e}")

    # Also save a summary CSV for easy comparison
    summary_csv_path = os.path.join(predicted_dataset_path, "timing_summary.csv")
    try:
        import csv
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Model Path', timing_stats['model_path']])
            writer.writerow(['Initialization Time (s)', f"{timing_stats['initialization_time']:.3f}"])
            writer.writerow(['Warmup Time (s)', f"{timing_stats['warmup_time']:.3f}"])
            writer.writerow(['Total Prediction Time (s)', f"{total_processing_time:.3f}"])
            writer.writerow(['Files Processed', processed_files])
            writer.writerow(['Total Characters', total_chars])
            writer.writerow(['Total Entities', total_entities])
            if processed_files > 0:
                writer.writerow(['Avg Time Per File (s)', f"{timing_stats['avg_time_per_file']:.4f}"])
                writer.writerow(['Throughput Files/s', f"{timing_stats['throughput_files_per_sec']:.2f}"])
                writer.writerow(['Throughput Chars/s', f"{timing_stats['throughput_chars_per_sec']:.0f}"])
                writer.writerow(['Avg Entities Per File', f"{timing_stats['avg_entities_per_file']:.2f}"])
        print(f"💾 Summary CSV saved to: {summary_csv_path}")
    except Exception as e:
        print(f"⚠️  Failed to save summary CSV: {e}")

    print(f"\n" + "="*80)
    print(f"Processing completed!")
    print("="*80)

    return text2pred, dataset_path, predicted_dataset_path, model_entity_types


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Binder NER batch inference with configurable encodings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default UTF-8 encoding
  python text2ner.py

  # Read and write cp1251 (Windows Cyrillic)
  python text2ner.py --input-encoding cp1251 --output-encoding cp1251

  # Read cp1251, write UTF-8
  python text2ner.py --input-encoding cp1251 --output-encoding utf-8

  # Other encodings
  python text2ner.py --input-encoding windows-1251 --output-encoding utf-8
        '''
    )
    parser.add_argument(
        '--input-encoding',
        type=str,
        default='utf-8',
        help='Encoding for reading .txt and .ann files (default: utf-8). Common: cp1251, windows-1251, latin-1'
    )
    parser.add_argument(
        '--output-encoding',
        type=str,
        default='utf-8',
        help='Encoding for writing .txt and .ann files (default: utf-8). Common: cp1251, windows-1251, latin-1'
    )

    args = parser.parse_args()

    # Run the efficient streaming batch prediction demo
    text2pred, dataset_path, predicted_dataset_path, model_entity_types = stream_batch_prediction_demo(
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding
    )

    # Original evaluation logic (kept for compatibility)
    print(f"\n Loading gold annotations from: {dataset_path}")
    print(f"Filtering to {len(model_entity_types)} entity types: {sorted(model_entity_types)}")
    
    # Load gold annotations
    for root, dirs, files in os.walk(dataset_path):
        for f in tqdm(sorted(files), desc="Loading annotations"):
            if f.endswith(".ann"):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8", newline = "") as annfile:
                        golds = []
                        for line in annfile:
                            line_tokens = line.split()
                            if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                                try:
                                    entity_type = line_tokens[1]
                                    # Only include entities with types the model was trained on
                                    if entity_type in model_entity_types:
                                        tags.add(entity_type)
                                        start_char = int(line_tokens[2])
                                        end_char = int(line_tokens[3])
                                        golds.append((start_char, end_char, entity_type))
                                except ValueError:
                                    continue
                    
                    txtf = f.replace('.ann', '.txt')
                    if txtf in text2pred:
                        text = text2pred[txtf]["text"]
                        text2pred[txtf]["golds"] = [(s, e, t, text[s:e]) for s, e, t in golds]
                    
                except Exception as e:
                    print(f"  Error reading annotation file {f}: {e}")
                    traceback.print_exc()

    # print(list(text2pred.values())[0])

    def compute_tp_fn_fp(predictions, labels):
        # tp, fn, fp
        if len(predictions) == 0:
            return {"tp": 0, "fn": len(labels), "fp": 0}
        if len(labels) == 0:
            return {"tp": 0, "fn": 0, "fp": len(predictions)}
        tp = len(predictions & labels)
        fn = len(labels) - tp
        fp = len(predictions) - tp
        return {"tp": tp, "fn": fn, "fp": fp}


    def compute_precision_recall_f1(tp, fn, fp):
        if tp + fp + fn == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if tp + fp == 0:
            return {"precision": 0.0, "recall": .0, "f1": .0}
        if tp + fn == 0:
            return {"precision": .0, "recall": 0.0, "f1": .0}
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        return {"precision": precision, "recall": recall, "f1": f1}

    metrics = {}
    for tag in tags:
        metrics[tag] = {"tp" : 0, "fp" : 0, "fn" : 0}.copy()

    metrics["total"] = {"tp" : 0, "fp" : 0, "fn" : 0}.copy()
    # --- NEW SECTION: Save Predictions FIRST (Outside Evaluation Loop) ---
    print(f"\nSaving predictions to: {predicted_dataset_path}")
    # Create output directory again (ensure it exists)
    os.makedirs(predicted_dataset_path, exist_ok=True)

    # Iterate through all processed files and save their text and predictions
    for filename, data in text2pred.items():
        # Save the original text
        text_file_path = os.path.join(predicted_dataset_path, filename)
        with open(text_file_path, "w", encoding=args.output_encoding) as tf:
            tf.write(data["text"])

        # Save the predictions in .ann format
        ann_filename = filename.replace('.txt', '.ann')
        ann_file_path = os.path.join(predicted_dataset_path, ann_filename)
        with open(ann_file_path, "w", encoding=args.output_encoding) as af:
            for p_idx, p in enumerate(data["preds"]):
                # Handle both (s,e,t) and (s,e,t,text) formats from the model output
                if len(p) == 3:
                    s, e, t = p
                    # Extract text span from the original text if not provided by the model
                    sp = data["text"][s:e]
                else: # Assume it's (s,e,t,text)
                    s, e, t, sp = p
                # Write the .ann format line
                af.write(f"T{p_idx + 1}\t{t} {s} {e}\t{sp}\n")
    print(f"Predictions saved successfully for {len(text2pred)} files.")
    # --- END NEW SECTION ---

    # Original evaluation logic (kept for compatibility) - now runs *after* saving predictions
    print(f"\nLoading gold annotations from: {dataset_path}")
    print(f"Filtering to {len(model_entity_types)} entity types: {sorted(model_entity_types)}")
    # Load gold annotations
    gold_files_found = False
    for root, dirs, files in os.walk(dataset_path):
        for f in tqdm(sorted(files), desc="Loading annotations"):
            if f.endswith(".ann"):
                gold_files_found = True
                try:
                    ann_content = read_file_with_fallback(os.path.join(root, f), args.input_encoding)
                    golds = []
                    for line in ann_content.splitlines():
                        line_tokens = line.split()
                        if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                            try:
                                entity_type = line_tokens[1]
                                # Only include entities with types the model was trained on
                                if entity_type in model_entity_types:
                                    tags.add(entity_type)
                                    start_char = int(line_tokens[2])
                                    end_char = int(line_tokens[3])
                                    golds.append((start_char, end_char, entity_type))
                            except ValueError:
                                continue
                    txtf = f.replace('.ann', '.txt')
                    if txtf in text2pred:
                        text = text2pred[txtf]["text"]
                        text2pred[txtf]["golds"] = [(s, e, t, text[s:e]) for s, e, t in golds]
                except Exception as e:
                    print(f"  Error reading annotation file {f}: {e}")
                    traceback.print_exc()

    if not gold_files_found:
        print("  No .ann files found in the dataset path. Skipping evaluation metrics.")
    else:
        print("  Gold annotations loaded.")

    # DEBUG: Show entity type distributions
    print(f"\n" + "="*80)
    print(f"DEBUG: ENTITY TYPE ANALYSIS")
    print("="*80)
    gold_type_counts = {}
    pred_type_counts = {}
    for filename, data in text2pred.items():
        for pred in data['preds']:
            entity_type = pred[2] if len(pred) >= 3 else "UNKNOWN"
            pred_type_counts[entity_type] = pred_type_counts.get(entity_type, 0) + 1
        for gold in data.get('golds', []):
            entity_type = gold[2] if len(gold) >= 3 else "UNKNOWN"
            gold_type_counts[entity_type] = gold_type_counts.get(entity_type, 0) + 1

    print(f"\nTotal entity types in model: {len(model_entity_types)}")
    print(f"Entity types in gold annotations: {len(gold_type_counts)}")
    print(f"Entity types in predictions: {len(pred_type_counts)}")
    print(f"\nGold entity type counts: {dict(sorted(gold_type_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
    print(f"Pred entity type counts: {dict(sorted(pred_type_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")

    # DEBUG: Show sample gold annotations vs predictions
    print(f"\n" + "="*80)
    print(f"DEBUG: GOLD vs PREDICTIONS COMPARISON")
    print("="*80)
    sample_files = [f for f in list(text2pred.keys())[:3] if 'golds' in text2pred[f]]
    for filename in sample_files:
        data = text2pred[filename]
        print(f"\nFile: {filename}")
        print(f"Gold annotations count: {len(data.get('golds', []))}")
        print(f"Predictions count: {len(data['preds'])}")

        if data.get('golds'):
            print(f"Sample gold annotations (first 5):")
            for i, gold in enumerate(data['golds'][:5]):
                if len(gold) >= 4:
                    s, e, t, text_span = gold[:4]
                else:
                    s, e, t = gold
                    text_span = "N/A"
                print(f"  {i+1}. [{s}:{e}] {t} = '{text_span}'")

        if data['preds']:
            print(f"Sample predictions (first 5):")
            for i, pred in enumerate(data['preds'][:5]):
                if len(pred) >= 4:
                    s, e, t, text_span = pred[:4]
                elif len(pred) == 3:
                    s, e, t = pred
                    text_span = data['text'][s:e]
                else:
                    text_span = "unknown"
                print(f"  {i+1}. [{s}:{e}] {t} = '{text_span}'")

    # print(list(text2pred.values())[0])

    metrics = {}
    for tag in tags:
        metrics[tag] = {"tp" : 0, "fp" : 0, "fn" : 0}.copy()
    metrics["total"] = {"tp" : 0, "fp" : 0, "fn" : 0}.copy()

    for file in text2pred.keys():
        # print(text2pred[file])
        # print(file)
        try:
            # Convert to (start,end,type) triples for comparison to avoid textual mismatches
            golds = set((s, e, t) for (s, e, t, *_unused) in text2pred[file].get("golds", [])) # Use .get() to handle missing 'golds'
            preds = set((s, e, t) for (s, e, t, *_unused) in text2pred[file]["preds"])
        except KeyError:
            print(file)
            continue
        # print(sorted(list(golds)))
        # print(sorted(list(preds)))

        # The saving of .txt and .ann files was moved to the new section above
        # This loop now only calculates metrics if 'golds' exist
        if "golds" in text2pred[file]: # Only calculate metrics if gold annotations were loaded for this file
            ts = compute_tp_fn_fp(preds, golds)
            tp, fn, fp = ts["tp"], ts["fn"], ts["fp"]
            metrics["total"]["tp"] += tp
            metrics["total"]["fp"] += fp
            metrics["total"]["fn"] += fn
            for tag in tags:
                # Convert back to original format for tag filtering
                original_golds = text2pred[file]["golds"]
                original_preds = text2pred[file]["preds"]
                tagged_golds = set((s, e, t) for (s, e, t, *_unused) in original_golds if t == tag)
                tagged_preds = set((s, e, t) for (s, e, t, *_unused) in original_preds if t == tag)
                tagged_ts = compute_tp_fn_fp(tagged_preds, tagged_golds)
                tagged_tp, tagged_fp, tagged_fn = tagged_ts["tp"], tagged_ts["fp"], tagged_ts["fn"]
                metrics[tag]["tp"] += tagged_tp
                metrics[tag]["fp"] += tagged_fp
                metrics[tag]["fn"] += tagged_fn
        else:
            # If no golds, this file contributes 0 to TP/FN, and its preds are FP for their respective tags
            # Note: This logic assumes tags are known from gold files loaded earlier.
            # If *no* gold files exist at all, 'tags' set will be empty, and per-tag metrics won't be calculated later.
            # This loop primarily handles files where golds exist for *some* types but maybe not all.
            # For the total metrics, if no golds exist for a file, its preds are FPs.
            # The total calculation loop below handles this implicitly by summing over files with golds.
            # If a file has no golds, its preds contribute FP to the total count.
            # However, to correctly update per-tag FP counts for files without golds:
            for s, e, t, *_unused in text2pred[file]["preds"]:
                 if t in metrics: # Check if tag exists in metrics dict (loaded from gold)
                     metrics[t]["fp"] += 1 # If golds exist for this tag anywhere, but not in this specific file, its a FP
            # The total metrics calculation will be based on the files that *do* have golds loaded.
            # If *no* files have golds, the 'total' metrics will remain {tp:0, fp:0, fn:0}.


    metrics["total"] = {**metrics["total"], **compute_precision_recall_f1(metrics["total"]["tp"], metrics["total"]["fn"], metrics["total"]["fp"])}
    for tag in tags:
        metrics[tag] = {**metrics[tag], **compute_precision_recall_f1(metrics[tag]["tp"], metrics[tag]["fn"], metrics[tag]["fp"])}

    # Removed verbose detailed error analysis for cleaner output
    # Uncomment the section below if you need detailed debugging information

    # print(metrics)
    #
    # # Detailed error analysis for problematic classes
    # print("\n" + "="*80)
    # print("DETAILED ERROR ANALYSIS")
    # print("="*80)
    # ...

    # Calculate macro averages (average of per-class metrics)
    macro_precision = sum(metrics[tag]["precision"] for tag in tags) / len(tags) if tags else 0
    macro_recall = sum(metrics[tag]["recall"] for tag in tags) / len(tags) if tags else 0
    macro_f1 = sum(metrics[tag]["f1"] for tag in tags) / len(tags) if tags else 0

    # Micro averages are the same as total (already calculated from all TP/FP/FN)
    micro_precision = metrics['total']['precision']
    micro_recall = metrics['total']['recall']
    micro_f1 = metrics['total']['f1']

    print(f"\n" + "="*80)
    print("SAVING PERFORMANCE METRICS")
    print("="*80)

    # Save detailed per-class metrics to CSV
    metrics_csv_path = os.path.join(predicted_dataset_path, "performance_metrics.csv")
    try:
        with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'])

            # Per-class metrics (sorted alphabetically)
            for tag in sorted(tags):
                writer.writerow([
                    tag,
                    metrics[tag]['tp'],
                    metrics[tag]['fp'],
                    metrics[tag]['fn'],
                    f"{metrics[tag]['precision']:.4f}",
                    f"{metrics[tag]['recall']:.4f}",
                    f"{metrics[tag]['f1']:.4f}"
                ])

            # Empty row separator
            writer.writerow([])

            # Micro average (total)
            writer.writerow([
                'Micro Average',
                metrics['total']['tp'],
                metrics['total']['fp'],
                metrics['total']['fn'],
                f"{micro_precision:.4f}",
                f"{micro_recall:.4f}",
                f"{micro_f1:.4f}"
            ])

            # Macro average
            writer.writerow([
                'Macro Average',
                '',  # No TP for macro
                '',  # No FP for macro
                '',  # No FN for macro
                f"{macro_precision:.4f}",
                f"{macro_recall:.4f}",
                f"{macro_f1:.4f}"
            ])

        print(f"📊 Performance metrics saved to: {metrics_csv_path}")
    except Exception as e:
        print(f"⚠️  Failed to save performance metrics: {e}")

    # Save summary metrics (easy to copy-paste)
    summary_metrics_path = os.path.join(predicted_dataset_path, "metrics_summary.csv")
    try:
        with open(summary_metrics_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Micro Precision', f"{micro_precision:.4f}"])
            writer.writerow(['Micro Recall', f"{micro_recall:.4f}"])
            writer.writerow(['Micro F1', f"{micro_f1:.4f}"])
            writer.writerow(['Macro Precision', f"{macro_precision:.4f}"])
            writer.writerow(['Macro Recall', f"{macro_recall:.4f}"])
            writer.writerow(['Macro F1', f"{macro_f1:.4f}"])
            writer.writerow(['Total TP', metrics['total']['tp']])
            writer.writerow(['Total FP', metrics['total']['fp']])
            writer.writerow(['Total FN', metrics['total']['fn']])
            writer.writerow(['Number of Classes', len(tags)])

        print(f"📊 Summary metrics saved to: {summary_metrics_path}")
    except Exception as e:
        print(f"⚠️  Failed to save summary metrics: {e}")

    # Print summary for easy viewing
    print(f"\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\nMicro Averages (overall):")
    print(f"  Precision: {micro_precision:.4f}")
    print(f"  Recall:    {micro_recall:.4f}")
    print(f"  F1:        {micro_f1:.4f}")

    print(f"\nMacro Averages (per-class average):")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1:        {macro_f1:.4f}")

    print(f"\nTotal Counts:")
    print(f"  TP: {metrics['total']['tp']}")
    print(f"  FP: {metrics['total']['fp']}")
    print(f"  FN: {metrics['total']['fn']}")
    print(f"  Classes: {len(tags)}")

    print("="*80)


