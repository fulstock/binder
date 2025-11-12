#!/usr/bin/env python3
"""
Consolidated BinderInference class for named entity recognition.
This module provides a unified interface for Binder model inference.
"""

import os
import sys
import gc
import re
import torch
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)

import nltk
from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer

from .config import BinderConfig
from .model import Binder
from .trainer import BinderDataCollator, BinderTrainer
from . import utils as postprocess_utils


@dataclass
class ModelArguments:
    """Arguments for Binder model configuration."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    binder_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Binder model from huggingface.co/models or local path to a saved Binder module."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
    hidden_dropout_prob: float = field(default=0.1)
    use_span_width_embedding: bool = field(default=True)
    linear_size: int = field(default=1024)
    init_temperature: float = field(default=2.0)
    start_loss_weight: float = field(default=0.25)
    end_loss_weight: float = field(default=0.25)
    span_loss_weight: float = field(default=0.25)
    threshold_loss_weight: float = field(default=0.25)
    ner_loss_weight: float = field(default=1.0)


@dataclass
class DataTrainingArguments:
    """Arguments for data processing and training configuration."""
    dataset_name: str = field(metadata={"help": "The name of the dataset to use."})
    entity_type_file: str = field(metadata={"help": "A json file for the entity types."})
    entity_type_key_field: str = field(default="name", metadata={"help": "The key field for entity type."})
    entity_type_desc_field: str = field(default="description", metadata={"help": "The description field for entity type."})
    dataset_entity_types: Optional[List[str]] = field(
        default=None, metadata={"help": "Entity types to include, leave empty to include all"}
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
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_span_length: int = field(default=10, metadata={"help": "Maximum span length for NER entities"})
    do_neutral_spans: bool = field(default=False, metadata={"help": "Whether to include neutral spans"})
    neutral_relative_threshold: float = field(default=0.5, metadata={"help": "Threshold for neutral spans"})
    prediction_postprocess_func: str = field(
        default="postprocess_nested_predictions",
        metadata={"help": "Post-processing function name"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "Path to training file"})


class SafeWordTokenizer:
    """Safe word tokenizer with timeout protection and fallbacks."""

    def __init__(self):
        self.nltk_tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize NLTK tokenizer with fallback."""
        try:
            self.nltk_tokenizer = NLTKWordTokenizer()
        except Exception as e:
            print(f"Failed to initialize NLTK tokenizer: {e}. Using regex fallback.")
            self.nltk_tokenizer = None

    def span_tokenize(self, text):
        """Tokenize text with fallback to regex."""
        if self.nltk_tokenizer:
            try:
                return list(self.nltk_tokenizer.span_tokenize(text))
            except Exception:
                pass

        # Regex fallback
        import re
        spans = []
        for match in re.finditer(r'\S+', text):
            spans.append((match.start(), match.end()))
        return spans


class BinderInference:
    """
    Consolidated BinderInference class for named entity recognition.

    This class provides a unified interface for loading and running inference
    with trained Binder models. It includes robust error handling, optimization
    features, and support for both single and batch predictions.
    """

    def __init__(self, config_path=None, device="auto", prediction_threshold_factor=1.0):
        """
        Initialize BinderInference with configuration and device settings.

        Args:
            config_path (str, optional): Path to inference configuration JSON file.
                If None, will search for config in ./inference/ directory.
            device (str): Device to use for inference ("auto", "cpu", "cuda").
            prediction_threshold_factor (float): Factor to adjust prediction threshold.
        """
        # Fix config path issue with flexible fallback
        if config_path is None:
            config_path = "./inference/inference-config.json"
            if not os.path.exists(config_path):
                config_path = "./inference/inference-config-optimized.json"
                if not os.path.exists(config_path):
                    config_path = "./conf/inference-config-optimized.json"
                    if not os.path.exists(config_path):
                        config_path = "./conf/lexical-conf.json"
                        if not os.path.exists(config_path):
                            raise FileNotFoundError(
                                "Config file not found. Please check that inference config exists in ./inference/ or ./conf/ directory"
                            )

        # Environment setup
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"Loading config from: {config_path}")
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=config_path)

        # Device configuration
        if device == "cpu":
            training_args.use_cpu = True
        else:
            training_args.use_cpu = False

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        set_seed(training_args.seed)

        # Cache tokenizers to avoid reloading
        self._ru_tokenizer = None
        self._word_tokenizer = None

        # Initialize lazily
        self._tokenizer = None
        self._trainer = None
        self._entity_type_cache = None

        # Optimization flags
        self._batch_mode = False
        self._cached_features = {}

        # Set device optimization
        self.device = device
        if device != "cpu":
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

        # Prediction threshold factor
        self.prediction_threshold_factor = prediction_threshold_factor
        if prediction_threshold_factor != 1.0:
            print(f"Using custom prediction threshold factor: {prediction_threshold_factor}")

    def _optimize_model(self, model):
        """Apply only safe, proven optimizations for faster inference."""
        # Disable gradient checkpointing for inference (saves memory and computation)
        try:
            if hasattr(model, 'config'):
                model.config.gradient_checkpointing = False
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
        except Exception:
            pass

        # DISABLED OPTIMIZATIONS (were causing slowdowns):
        # - Type embedding caching: hash computation overhead > savings
        # - torch.compile(): compilation overhead not worth it for small batches
        # - FP16: can be slower on CPU or older GPUs
        # - Static padding: adds unnecessary computation
        # - torch.jit.script: incompatible with some Transformers operations

        return model

    @property
    def ru_tokenizer(self):
        """Lazy loading of Russian tokenizer with fallback."""
        if self._ru_tokenizer is None:
            try:
                self._ru_tokenizer = load("tokenizers/punkt/russian.pickle")
                print("Russian tokenizer loaded")
            except Exception as e:
                print(f"Failed to load Russian tokenizer: {e}. Using simple sentence splitting.")
                # Fallback sentence tokenizer
                class SimpleSentenceTokenizer:
                    def span_tokenize(self, text):
                        sentences = re.split(r'[.!?]+', text)
                        pos = 0
                        for sentence in sentences:
                            if sentence.strip():
                                start = text.find(sentence.strip(), pos)
                                if start != -1:
                                    end = start + len(sentence.strip())
                                    yield (start, end)
                                    pos = end
                self._ru_tokenizer = SimpleSentenceTokenizer()
        return self._ru_tokenizer

    @property
    def word_tokenizer(self):
        """Lazy loading of word tokenizer with timeout protection."""
        if self._word_tokenizer is None:
            try:
                self._word_tokenizer = SafeWordTokenizer()
                print("Word tokenizer loaded")
            except Exception as e:
                print(f"Failed to load word tokenizer: {e}. Using fallback.")
                self._word_tokenizer = SafeWordTokenizer()
        return self._word_tokenizer

    @property
    def tokenizer(self):
        """Lazy loading of main tokenizer."""
        if self._tokenizer is None:
            try:
                # print("Loading main tokenizer...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
                    cache_dir=self.model_args.cache_dir,
                    use_fast=True,
                    revision=self.model_args.model_revision,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                    add_prefix_space=True,
                )
                # print("Main tokenizer loaded")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {e}")
        return self._tokenizer

    @property
    def trainer(self):
        """Lazy loading of trainer."""
        if self._trainer is None:
            try:
                # print("Initializing trainer (this may take a few minutes)...")
                self._initialize_trainer()
                # print("Trainer initialized")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize trainer: {e}")
        return self._trainer

    def _initialize_trainer(self):
        """Initialize trainer with optimizations and safety checks."""
        tokenizer = self.tokenizer
        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args

        # Ensure DataLoader does not use multiprocessing workers
        if hasattr(training_args, "dataloader_num_workers") and training_args.dataloader_num_workers != 0:
            print(f"Overriding dataloader_num_workers from {training_args.dataloader_num_workers} to 0")
            training_args.dataloader_num_workers = 0

        # Disable persistent workers
        if hasattr(training_args, "dataloader_persistent_workers"):
            if training_args.dataloader_persistent_workers:
                print("Disabling dataloader_persistent_workers")
            training_args.dataloader_persistent_workers = False

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError("This script only works with fast tokenizers")

        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        self.max_seq_length = max_seq_length

        # Garbage collection before heavy operations
        gc.collect()

        # Load entity type knowledge
        if self._entity_type_cache is None:
            # print("Loading entity type knowledge...")
            try:
                entity_type_knowledge = load_dataset(
                    "json", data_files=data_args.entity_type_file, cache_dir=model_args.cache_dir
                )["train"]
                entity_type_knowledge = entity_type_knowledge.filter(
                    lambda example: (
                        example["dataset"] == data_args.dataset_name and (
                            len(data_args.dataset_entity_types) == 0 or
                            example[data_args.entity_type_key_field] in data_args.dataset_entity_types
                        )
                    )
                )
                entity_type_knowledge = entity_type_knowledge.sort(data_args.entity_type_key_field)
                self._entity_type_cache = entity_type_knowledge
                # print(f"Loaded {len(entity_type_knowledge)} entity types")
            except Exception as e:
                raise RuntimeError(f"Failed to load entity type file '{data_args.entity_type_file}': {e}")

        entity_type_knowledge = self._entity_type_cache
        entity_type_id_to_str = [et[data_args.entity_type_key_field] for et in entity_type_knowledge]

        def prepare_type_features(examples):
            tokenized_examples = tokenizer(
                examples[data_args.entity_type_desc_field],
                truncation=True,
                max_length=max_seq_length,
                padding="longest" if len(entity_type_knowledge) <= 1000 else "max_length",
            )
            return tokenized_examples

        # print("Tokenizing entity type descriptions...")
        with training_args.main_process_first(desc="Tokenizing entity type descriptions"):
            tokenized_descriptions = entity_type_knowledge.map(
                prepare_type_features,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on type descriptions",
                remove_columns=entity_type_knowledge.column_names,
            )

        # Garbage collection before model loading
        gc.collect()

        # Initialize model with comprehensive error handling
        # print("Loading Binder model...")
        # print(f"Binder model path: {model_args.binder_model_name_or_path}")
        # print(f"Base model path: {model_args.model_name_or_path}")

        # Load config with fallback strategies
        try:
            from transformers import AutoConfig
            repository_config = AutoConfig.from_pretrained(
                model_args.binder_model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            print(f"Loaded config from repository: {model_args.binder_model_name_or_path}")

            # Create BinderConfig using the repository config as base
            config = BinderConfig(
                pretrained_model_name_or_path=repository_config.pretrained_model_name_or_path or model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                hidden_dropout_prob=getattr(repository_config, 'hidden_dropout_prob', model_args.hidden_dropout_prob),
                max_span_width=getattr(repository_config, 'max_span_width', data_args.max_seq_length + 1),
                use_span_width_embedding=getattr(repository_config, 'use_span_width_embedding', model_args.use_span_width_embedding),
                do_neutral_spans=getattr(repository_config, 'do_neutral_spans', data_args.do_neutral_spans),
                linear_size=getattr(repository_config, 'linear_size', model_args.linear_size),
                init_temperature=getattr(repository_config, 'init_temperature', model_args.init_temperature),
                start_loss_weight=getattr(repository_config, 'start_loss_weight', model_args.start_loss_weight),
                end_loss_weight=getattr(repository_config, 'end_loss_weight', model_args.end_loss_weight),
                span_loss_weight=getattr(repository_config, 'span_loss_weight', model_args.span_loss_weight),
                threshold_loss_weight=getattr(repository_config, 'threshold_loss_weight', model_args.threshold_loss_weight),
                ner_loss_weight=getattr(repository_config, 'ner_loss_weight', model_args.ner_loss_weight),
            )

        except Exception as config_error:
            print(f"Could not load repository config: {config_error}")
            print("Falling back to local config creation")

            config = BinderConfig(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                hidden_dropout_prob=model_args.hidden_dropout_prob,
                max_span_width=data_args.max_seq_length + 1,
                use_span_width_embedding=model_args.use_span_width_embedding,
                do_neutral_spans=data_args.do_neutral_spans,
                linear_size=model_args.linear_size,
                init_temperature=model_args.init_temperature,
                start_loss_weight=model_args.start_loss_weight,
                end_loss_weight=model_args.end_loss_weight,
                span_loss_weight=model_args.span_loss_weight,
                threshold_loss_weight=model_args.threshold_loss_weight,
                ner_loss_weight=model_args.ner_loss_weight,
            )

        # Load model with multiple fallback strategies
        def load_model_with_workarounds():
            """Load model with comprehensive error handling."""
            # Strategy 1: Direct loading
            try:
                # print("Strategy 1: Direct model loading...")
                model = Binder.from_pretrained(model_args.binder_model_name_or_path, config=config)
                # print("Strategy 1 successful: Model loaded directly")
                return model
            except Exception as e1:
                pass  # print(f"Strategy 1 failed: {e1}")

            # Strategy 2: Load with trust_remote_code
            try:
                # print("Strategy 2: Loading with trust_remote_code...")
                model = Binder.from_pretrained(
                    model_args.binder_model_name_or_path,
                    config=config,
                    trust_remote_code=True
                )
                # print("Strategy 2 successful: Model loaded with trust_remote_code")
                return model
            except Exception as e2:
                pass  # print(f"Strategy 2 failed: {e2}")

            # Strategy 3: Force download
            try:
                # print("Strategy 3: Force download to avoid cache corruption...")
                model = Binder.from_pretrained(
                    model_args.binder_model_name_or_path,
                    config=config,
                    force_download=True,
                    resume_download=True
                )
                # print("Strategy 3 successful: Model loaded with force download")
                return model
            except Exception as e3:
                pass  # print(f"Strategy 3 failed: {e3}")

            raise RuntimeError(f"Cannot load model from {model_args.binder_model_name_or_path}. All strategies failed.")

        # Execute model loading
        try:
            model = load_model_with_workarounds()
            # print("Model loading completed successfully")
        except Exception as final_error:
            print(f"All model loading strategies failed: {final_error}")
            raise RuntimeError("Model loading failed completely")

        # Configure model
        try:
            model.eval()
            # print("Model set to evaluation mode")
        except Exception as eval_error:
            print(f"Could not set model to eval mode: {eval_error}")

        # Move to device
        if self.device != "cpu" and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).total_memory < 2 * 1024**3:
                print("⚠️  Limited GPU memory detected. Consider using CPU mode.")
            model = model.cuda()
            # print("Model moved to GPU")
        else:
            pass  # print("Using CPU for inference")

        # Apply optimizations
        model = self._optimize_model(model)

        # Ensure trainer doesn't remove columns
        if hasattr(training_args, "remove_unused_columns") and training_args.remove_unused_columns:
            training_args.remove_unused_columns = False

        # Data collator
        data_collator = BinderDataCollator(
            type_input_ids=tokenized_descriptions["input_ids"],
            type_attention_mask=tokenized_descriptions["attention_mask"],
            type_token_type_ids=tokenized_descriptions["token_type_ids"] if "token_type_ids" in tokenized_descriptions else None,
        )

        # Post-processing function
        def post_processing_function(examples, features, predictions, stage="eval"):
            # Apply threshold factor if needed
            if self.prediction_threshold_factor != 1.0:
                # Custom post-processing with threshold factor
                try:
                    from .utils import postprocess_nested_predictions_with_threshold
                    metrics = postprocess_nested_predictions_with_threshold(
                        examples=examples,
                        features=features,
                        predictions=predictions,
                        id_to_type=entity_type_id_to_str,
                        max_span_length=data_args.max_span_length,
                        output_dir=training_args.output_dir if training_args.should_save else None,
                        prefix=stage,
                        neutral_relative_threshold=data_args.neutral_relative_threshold,
                        tokenizer=tokenizer,
                        train_file=data_args.train_file,
                        threshold_factor=self.prediction_threshold_factor,
                    )
                except (ImportError, AttributeError):
                    # Fallback to original function
                    metrics = getattr(postprocess_utils, data_args.prediction_postprocess_func)(
                        examples=examples,
                        features=features,
                        predictions=predictions,
                        id_to_type=entity_type_id_to_str,
                        max_span_length=data_args.max_span_length,
                        output_dir=training_args.output_dir if training_args.should_save else None,
                        prefix=stage,
                        neutral_relative_threshold=data_args.neutral_relative_threshold,
                        tokenizer=tokenizer,
                        train_file=data_args.train_file,
                    )
            else:
                # Standard post-processing
                metrics = getattr(postprocess_utils, data_args.prediction_postprocess_func)(
                    examples=examples,
                    features=features,
                    predictions=predictions,
                    id_to_type=entity_type_id_to_str,
                    max_span_length=data_args.max_span_length,
                    output_dir=training_args.output_dir if training_args.should_save else None,
                    prefix=stage,
                    neutral_relative_threshold=data_args.neutral_relative_threshold,
                    tokenizer=tokenizer,
                    train_file=data_args.train_file,
                )
            return metrics

        # Initialize trainer
        print("Creating trainer...")
        self._trainer = BinderTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            eval_examples=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=None,
        )

    def _tokenize_text_optimized(self, text):
        """Optimized text tokenization with caching."""
        text_hash = hash(text)
        if text_hash in self._cached_features:
            return self._cached_features[text_hash]

        try:
            sentence_spans = list(self.ru_tokenizer.span_tokenize(text))

            offset_mapping = []
            for span in sentence_spans:
                start, end = span
                context = text[start:end]
                word_spans = list(self.word_tokenizer.span_tokenize(context))
                offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

        except Exception as e:
            print(f"Tokenization error: {e}. Using simple fallback.")
            # Simple fallback
            words = text.split()
            offset_mapping = []
            pos = 0
            for word in words:
                start = text.find(word, pos)
                if start != -1:
                    end = start + len(word)
                    offset_mapping.append((start, end))
                    pos = end

        if offset_mapping:
            start_words, end_words = zip(*offset_mapping)
        else:
            start_words, end_words = (), ()

        result = (start_words, end_words)

        # Cache result
        if len(self._cached_features) < 1000:
            self._cached_features[text_hash] = result

        return result

    def predict(self, text):
        """Perform named entity recognition on input text."""
        return self._predict_single(text)

    def _predict_single(self, text):
        """Internal method for single text prediction."""
        trainer = self.trainer
        training_args = self.training_args
        data_args = self.data_args

        start_words, end_words = self._tokenize_text_optimized(text)

        predict_examples = Dataset.from_dict({
            "text": [text],
            "id": [0],
            "word_start_chars": [start_words],
            "word_end_chars": [end_words],
            "entity_types": [[]],
            "entity_start_chars": [[]],
            "entity_end_chars": [[]]
        })

        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))

        # Force consistent padding
        original_pad_setting = data_args.pad_to_max_length
        data_args.pad_to_max_length = True

        try:
            # Create prediction dataset
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_examples.map(
                    lambda x: self._prepare_validation_features(x, "test"),
                    batched=True,
                    remove_columns=predict_examples.column_names,
                    desc="Processing dataset features"
                )

            # Run inference
            with torch.no_grad():
                results = trainer.predict(
                    predict_dataset,
                    predict_examples,
                    ignore_keys=["offset_mapping", "example_id", "split", "token_start_mask", "token_end_mask"]
                )
        finally:
            # Restore original padding setting
            data_args.pad_to_max_length = original_pad_setting

        # Process predictions
        predictions = results.predictions
        processed_predictions = []
        for p in predictions:
            processed_predictions.append((p.start_char, p.end_char, p.entity_type, p.text))
        processed_predictions = sorted(processed_predictions, key=lambda x: (x[0], x[1]))

        return processed_predictions

    def predict_batch(self, texts):
        """Perform batch prediction on multiple texts."""
        if not isinstance(texts, (list, tuple)):
            texts = [texts]

        if len(texts) == 1:
            return [self._predict_single(texts[0])]

        # Batch processing implementation
        trainer = self.trainer
        training_args = self.training_args
        data_args = self.data_args

        # Tokenize all texts
        all_start_words = []
        all_end_words = []
        for text in texts:
            start_words, end_words = self._tokenize_text_optimized(text)
            all_start_words.append(start_words)
            all_end_words.append(end_words)

        # Create batch dataset
        predict_examples = Dataset.from_dict({
            "text": texts,
            "id": list(range(len(texts))),
            "word_start_chars": all_start_words,
            "word_end_chars": all_end_words,
            "entity_types": [[] for _ in texts],
            "entity_start_chars": [[] for _ in texts],
            "entity_end_chars": [[] for _ in texts]
        })

        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(min(len(texts), data_args.max_predict_samples)))

        # Force consistent padding
        original_pad_setting = data_args.pad_to_max_length
        data_args.pad_to_max_length = True

        try:
            # Create prediction dataset
            with training_args.main_process_first(desc="batch prediction dataset map pre-processing"):
                predict_dataset = predict_examples.map(
                    lambda x: self._prepare_validation_features(x, "test"),
                    batched=True,
                    remove_columns=predict_examples.column_names,
                    desc="Processing dataset features"
                )

            # Run inference
            with torch.no_grad():
                results = trainer.predict(
                    predict_dataset,
                    predict_examples,
                    ignore_keys=["offset_mapping", "example_id", "split", "token_start_mask", "token_end_mask"]
                )
        finally:
            # Restore original padding setting
            data_args.pad_to_max_length = original_pad_setting

        # Process predictions
        predictions = results.predictions

        # Group predictions by text ID
        text_predictions = [[] for _ in texts]
        id_to_text_index = {str(i): i for i in range(len(texts))}

        for p in predictions:
            text_id_str = str(p.id)
            if text_id_str in id_to_text_index:
                text_index = id_to_text_index[text_id_str]
                text_predictions[text_index].append((p.start_char, p.end_char, p.entity_type, p.text))

        # Sort predictions for each text
        for i in range(len(text_predictions)):
            text_predictions[i] = sorted(text_predictions[i], key=lambda x: (x[0], x[1]))

        return text_predictions

    def _prepare_validation_features(self, examples, split):
        """Prepare features for validation/prediction."""
        tokenizer = self.tokenizer
        data_args = self.data_args
        max_seq_length = self.max_seq_length

        tokenized_examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["split"] = []
        tokenized_examples["example_id"] = []
        tokenized_examples["token_start_mask"] = []
        tokenized_examples["token_end_mask"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            tokenized_examples["split"].append(split)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Create token masks
            token_start_mask, token_end_mask = [], []
            word_start_chars = examples["word_start_chars"][sample_index]
            word_end_chars = examples["word_end_chars"][sample_index]

            for index, (start_char, end_char) in enumerate(tokenized_examples["offset_mapping"][i]):
                if sequence_ids[index] != 0:
                    token_start_mask.append(0)
                    token_end_mask.append(0)
                else:
                    token_start_mask.append(int(start_char in word_start_chars))
                    token_end_mask.append(int(end_char in word_end_chars))

            tokenized_examples["token_start_mask"].append(token_start_mask)
            tokenized_examples["token_end_mask"].append(token_end_mask)

            # Set non-text offsets to None
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 0 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def warm_up(self, sample_text="Это тестовый текст для прогрева модели."):
        """Warm up the model with a sample prediction."""
        _ = self.predict(sample_text)

    def clear_cache(self):
        """Clear feature cache to free memory."""
        self._cached_features.clear()

    def enable_batch_mode(self):
        """Enable optimizations for batch processing."""
        self._batch_mode = True
        target_batch_size = self.training_args.per_device_eval_batch_size * 2
        print(f"Batch mode enabled. Setting batch size to: {target_batch_size}")
        self.training_args.per_device_eval_batch_size = target_batch_size

    def get_entity_types(self):
        """Get the list of supported entity types."""
        if self._entity_type_cache is not None:
            return [et[self.data_args.entity_type_key_field] for et in self._entity_type_cache]
        return []

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "model_path": self.model_args.binder_model_name_or_path,
            "base_model_path": self.model_args.model_name_or_path,
            "device": str(self.device),
            "entity_types": self.get_entity_types(),
            "max_seq_length": getattr(self, 'max_seq_length', self.data_args.max_seq_length),
            "num_entity_types": len(self.get_entity_types()),
            "prediction_threshold_factor": self.prediction_threshold_factor,
        }