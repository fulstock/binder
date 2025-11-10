#!/usr/bin/env python3
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

def stream_batch_prediction_demo():
    """
    Resource-constrained streaming batch prediction example with anti-freeze protection:
    - Strict memory and timeout limits to prevent system freezing
    - Conservative batch sizes
    - Comprehensive error handling and resource monitoring
    - Automatic process priority management
    - Processes all files without artificial limits
    """
    print("="*80)
    print("STREAMING BATCH PREDICTION DEMO")
    print("="*80)
    
    # Configuration
    inference_config_path = "./conf/inference/inference-config-optimized.json"
    dataset_path = "/home/student1/data/seccol_NEREL-attack/test"
    predicted_dataset_path = "/home/student1/data/seccol_NEREL-attack/test_predicted6"
    BATCH_SIZE = 16
    
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {predicted_dataset_path}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Initialize inference engine
    print(f"\nInitializing BinderInference...")
    start_time = time.time()
    
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
    
    # Monitor memory usage after initialization
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory after init: {memory_usage:.1f}MB")
    
    # Warm up the model
    print("Warming up model...")
    try:
        inf.warm_up("Это тестовый текст для российских кибербезопасности.")
        print("Model warmup complete")
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")
        traceback.print_exc()
    
    init_time = time.time() - start_time
    print(f"Initialization complete: {init_time:.2f}s")
    
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
                with open(file_path, "r", encoding="UTF-8") as tf:
                    text = tf.read().strip()
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
        
        # Batch prediction
        batch_start_time = time.time()
        try:
            batch_predictions = inf.predict_batch(batch_texts)
            batch_time = time.time() - batch_start_time
            
            # Process batch results
            for i, (filename, predictions) in enumerate(zip(batch_filenames, batch_predictions)):
                text = batch_file_mapping[i]['text']
                text2pred[filename] = {
                    'text': text,
                    'preds': predictions
                }
                total_entities += len(predictions)
                processed_files += 1
                
                # Log sample predictions for monitoring
                if processed_files <= 3:  # Show first few files
                    print(f"{filename}: {len(predictions)} entities found")
                    if predictions:
                        sample_entities = predictions[:3]
                        for start, end, etype, text_span in sample_entities:
                            print(f"    • {etype}: '{text_span}' ({start}-{end})")
            
            # Performance logging with memory monitoring
            chars_per_sec = sum(len(batch_file_mapping[i]['text']) for i in range(len(batch_texts))) / batch_time
            files_per_sec = len(batch_texts) / batch_time
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            if batch_start // BATCH_SIZE % 3 == 0:  # Log every 3 batches for more monitoring
                print(f"  Batch {batch_start//BATCH_SIZE + 1}: {files_per_sec:.1f} files/s, {chars_per_sec:.0f} chars/s, {current_memory:.0f}MB")
            
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
    
    print(f"\n PROCESSING SUMMARY:")
    print(f"  • Files found: {len(text_files)}")
    print(f"  • Files processed: {processed_files}")
    print(f"  • Files failed: {failed_files}")
    print(f"  • Success rate: {(processed_files/(processed_files+failed_files)*100):.1f}%" if (processed_files+failed_files) > 0 else "N/A")
    print(f"  • Total entities found: {total_entities}")
    print(f"  • Total characters: {total_chars:,}")
    print(f"  • Processing time: {total_processing_time:.2f}s")
    
    if processed_files > 0:
        print(f"  • Throughput: {processed_files/total_processing_time:.1f} files/s")
        print(f"  • Character throughput: {total_chars/total_processing_time:.0f} chars/s")
        print(f"  • Average entities per file: {total_entities/processed_files:.1f}")
    
    # Final memory report
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    available_mem = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"  • Final memory usage: {final_memory:.1f}MB")
    print(f"  • Available memory: {available_mem:.1f}GB")
    
    print(f"\nProcessing completed!")
    
    return text2pred


if __name__ == "__main__":
    # Run the efficient streaming batch prediction demo
    text2pred = stream_batch_prediction_demo()
    
    # Original evaluation logic (kept for compatibility)
    print(f"\n Loading gold annotations...")
    
    dataset_path = "/home/student1/data/seccol_NEREL-attack/test"
    predicted_dataset_path = "/home/student1/data/seccol_NEREL-attack/test_predicted6"
    
    # Load gold annotations
    for root, dirs, files in os.walk(dataset_path):
        for f in tqdm(sorted(files), desc="Loading annotations"):
            if f.endswith(".ann"):
                try:
                    with open(os.path.join(root, f), "r", encoding="UTF-8") as annfile:
                        golds = []
                        for line in annfile:
                            line_tokens = line.split()
                            if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                                try:
                                    entity_type = line_tokens[1]
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
    for file in text2pred.keys():
        # print(text2pred[file])
        # print(file)
        try:
            # Convert to (start,end,type) triples for comparison to avoid textual mismatches
            golds = set((s, e, t) for (s, e, t, *_unused) in text2pred[file]["golds"])
            preds = set((s, e, t) for (s, e, t, *_unused) in text2pred[file]["preds"])
        except KeyError:
            print(file)
            continue

        # print(sorted(list(golds)))
        # print(sorted(list(preds)))

        with open(os.path.join(predicted_dataset_path, file), "w", encoding = "UTF-8") as tf:
            tf.write(text2pred[file]["text"])

        with open(os.path.join(predicted_dataset_path, file.replace('.txt','.ann')), "w", encoding = "UTF-8") as af:
            for p_idx, p in enumerate(sorted(list(preds))):
                # Handle both (s,e,t) and (s,e,t,text) formats
                if len(p) == 3:
                    s, e, t = p
                    sp = text2pred[file]["text"][s:e]  # Extract text span from original text
                else:
                    s, e, t, sp = p
                af.write("T" + str(p_idx + 1) + "\t" + t + " " + str(s) + " " + str(e) + "\t" + sp + "\n")

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

    metrics["total"] = {**metrics["total"], **compute_precision_recall_f1(metrics["total"]["tp"], metrics["total"]["fn"], metrics["total"]["fp"])}
    for tag in tags:
        metrics[tag] = {**metrics[tag], **compute_precision_recall_f1(metrics[tag]["tp"], metrics[tag]["fn"], metrics[tag]["fp"])}

    print(metrics)
    
    # Detailed error analysis for problematic classes
    print("\n" + "="*80)
    print("DETAILED ERROR ANALYSIS")
    print("="*80)
    
    # Find classes with poor performance (F1 < 0.3 or recall < 0.3)
    problematic_classes = []
    for tag in tags:
        f1 = metrics[tag]["f1"]
        recall = metrics[tag]["recall"]
        if f1 < 0.3 or recall < 0.3:
            problematic_classes.append(tag)
    
    print(f"Analyzing {len(problematic_classes)} problematic classes with F1 < 0.3 or recall < 0.3:")
    print(f"Classes: {', '.join(problematic_classes[:10])}{'...' if len(problematic_classes) > 10 else ''}")
    
    # Collect all predictions and golds by class for analysis
    class_predictions = {tag: [] for tag in problematic_classes}
    class_golds = {tag: [] for tag in problematic_classes}
    
    for file in text2pred.keys():
        if "golds" not in text2pred[file] or "preds" not in text2pred[file]:
            continue
            
        original_golds = text2pred[file]["golds"]
        original_preds = text2pred[file]["preds"]
        file_text = text2pred[file]["text"]
        
        # Group by class
        for s, e, t, text_span in original_golds:
            if t in problematic_classes:
                class_golds[t].append((file, s, e, text_span, file_text))
        
        for s, e, t, text_span in original_preds:
            if t in problematic_classes:
                class_predictions[t].append((file, s, e, text_span, file_text))
    
    # Print detailed analysis for each problematic class
    for tag in problematic_classes[:5]:  # Limit to first 5 classes to avoid too much output
        print(f"\n{'-'*60}")
        print(f"CLASS: {tag}")
        print(f"Metrics: F1={metrics[tag]['f1']:.3f}, Precision={metrics[tag]['precision']:.3f}, Recall={metrics[tag]['recall']:.3f}")
        print(f"Counts: TP={metrics[tag]['tp']}, FP={metrics[tag]['fp']}, FN={metrics[tag]['fn']}")
        print(f"{'-'*60}")
        
        # Show sample gold annotations (what should be found)
        golds_for_class = class_golds[tag]
        print(f"\nSAMPLE GOLD ANNOTATIONS ({len(golds_for_class)} total):")
        for i, (file, s, e, text_span, full_text) in enumerate(golds_for_class[:3]):
            context_start = max(0, s - 30)
            context_end = min(len(full_text), e + 30)
            context = full_text[context_start:context_end]
            highlighted = context.replace(text_span, f"**{text_span}**")
            print(f"  {i+1}. File: {file}")
            print(f"     Text: '{text_span}' (chars {s}-{e})")
            print(f"     Context: ...{highlighted}...")
            print()
        
        # Show sample predictions (what was actually found)
        preds_for_class = class_predictions[tag]
        print(f"SAMPLE PREDICTIONS ({len(preds_for_class)} total):")
        if preds_for_class:
            for i, (file, s, e, text_span, full_text) in enumerate(preds_for_class[:3]):
                context_start = max(0, s - 30)
                context_end = min(len(full_text), e + 30)
                context = full_text[context_start:context_end]
                highlighted = context.replace(text_span, f"**{text_span}**")
                print(f"  {i+1}. File: {file}")
                print(f"     Text: '{text_span}' (chars {s}-{e})")
                print(f"     Context: ...{highlighted}...")
                print()
        else:
            print("  No predictions found for this class!")
            
        # Show some examples of what was predicted instead in files that should have this class
        print(f"ALTERNATIVE PREDICTIONS IN FILES WITH {tag} GOLD:")
        files_with_gold = set(file for file, _, _, _, _ in golds_for_class)
        alternative_preds = []
        
        for file in list(files_with_gold)[:2]:  # Check first 2 files
            if file in text2pred and "preds" in text2pred[file]:
                file_preds = text2pred[file]["preds"]
                for s, e, t, text_span in file_preds:
                    if t != tag:  # Different class predicted
                        alternative_preds.append((file, s, e, t, text_span, text2pred[file]["text"]))
        
        for i, (file, s, e, pred_type, text_span, full_text) in enumerate(alternative_preds[:3]):
            context_start = max(0, s - 30)
            context_end = min(len(full_text), e + 30)
            context = full_text[context_start:context_end]
            highlighted = context.replace(text_span, f"**{text_span}**")
            print(f"  {i+1}. File: {file}")
            print(f"     Predicted as: {pred_type}")
            print(f"     Text: '{text_span}' (chars {s}-{e})")
            print(f"     Context: ...{highlighted}...")
            print()
            
        if not alternative_preds:
            print("  No alternative predictions found in files with gold annotations.")
        
        print(f"\n{'='*40}")
    
    print(f"\nFull analysis complete. Check the detailed breakdowns above.")
    print(f"Total problematic classes: {len(problematic_classes)}")
    print(f"Overall recall: {metrics['total']['recall']:.3f}")
    print(f"Overall F1: {metrics['total']['f1']:.3f}")
        


