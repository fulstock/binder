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

# Cross-platform timeout handler
class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Run function with timeout - cross-platform implementation"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, it timed out
        raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

def check_memory_available(min_memory_gb=2):
    """Check if minimum memory is available"""
    available_memory = psutil.virtual_memory().available / (1024**3)  # Convert to GB
    if available_memory < min_memory_gb:
        print(f"Warning: Only {available_memory:.1f}GB available memory. Minimum {min_memory_gb}GB recommended.")
        return False
    return True

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

class BinderInference:

    def __init__(self, config_path = None, device = "auto", prediction_threshold_factor=1.0, max_memory_gb=128):
        # Check memory before initialization with configurable limit
        available_memory = psutil.virtual_memory().available / (1024**3)
        if available_memory < 2:
            print("Continuing with very low memory. System may become unstable.")
        elif available_memory > max_memory_gb:
            print(f"Available memory ({available_memory:.1f}GB) exceeds configured limit ({max_memory_gb}GB). Using conservative resource settings.")
        
        self.max_memory_gb = max_memory_gb
        print(f"Memory status: {available_memory:.1f}GB available, {max_memory_gb}GB limit configured")
        
        # Fix config path issue
        if config_path is None:
            config_path = "./inference/inference-config.json"
            if not os.path.exists(config_path):
                config_path = "./inference/inference-config-optimized.json"
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        "Config file not found. Please check that inference config exists in ./inference/ directory"
                    )
        
        os.environ["WANDB_DISABLED"] = "true"

        print(f"Loading config from: {config_path}")
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=config_path)
        
        # Reduce multiprocessing workers to prevent resource exhaustion
        # Ultra-conservative settings for stability on low-resource systems
        max_workers = 2  # Limit to 2 workers to prevent system overload
        
        if training_args.dataloader_num_workers > max_workers:
            print(f"Reducing dataloader workers from {training_args.dataloader_num_workers} to {max_workers} to prevent freezing")
            training_args.dataloader_num_workers = max_workers
        if data_args.preprocessing_num_workers and data_args.preprocessing_num_workers > max_workers:
            print(f"Reducing preprocessing workers from {data_args.preprocessing_num_workers} to {max_workers} to prevent freezing")
            data_args.preprocessing_num_workers = max_workers
            
        # Reduce batch sizes for memory efficiency
        if hasattr(training_args, 'per_device_eval_batch_size') and training_args.per_device_eval_batch_size > 4:
            print(f"Reducing batch size from {training_args.per_device_eval_batch_size} to 4 for memory efficiency")
            training_args.per_device_eval_batch_size = 4
        
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
        
        # Prediction threshold factor - can be lowered to find more entities
        self.prediction_threshold_factor = prediction_threshold_factor
        if prediction_threshold_factor != 1.0:
            print(f"Using custom prediction threshold factor: {prediction_threshold_factor}")
            
    @property
    def ru_tokenizer(self):
        """Lazy loading of Russian tokenizer"""
        if self._ru_tokenizer is None:
            def _load_ru_tokenizer():
                return load("tokenizers/punkt/russian.pickle")
            
            try:
                self._ru_tokenizer = run_with_timeout(_load_ru_tokenizer, 30)  # 30 second timeout
                print("Russian tokenizer loaded")
            except (TimeoutException, Exception) as e:
                print(f"Failed to load Russian tokenizer: {e}. Using simple sentence splitting.")
                # Fallback sentence tokenizer
                class SimpleSentenceTokenizer:
                    def span_tokenize(self, text):
                        # Simple sentence splitting on periods, exclamation marks, question marks
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
        """Lazy loading of word tokenizer with timeout"""
        if self._word_tokenizer is None:
            def _load_word_tokenizer():
                return SafeWordTokenizer()
            
            try:
                self._word_tokenizer = run_with_timeout(_load_word_tokenizer, 10)  # 10 second timeout
                print("Word tokenizer loaded")
            except (TimeoutException, Exception) as e:
                print(f"Failed to load word tokenizer: {e}. Using fallback.")
                self._word_tokenizer = SafeWordTokenizer()  # Should always work with regex fallback
        return self._word_tokenizer
    
    @property
    def tokenizer(self):
        """Lazy loading of main tokenizer with timeout"""
        if self._tokenizer is None:
            def _load_main_tokenizer():
                print("Loading main tokenizer...")
                return AutoTokenizer.from_pretrained(
                    self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
                    cache_dir=self.model_args.cache_dir,
                    use_fast=True,
                    revision=self.model_args.model_revision,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                    add_prefix_space=True,
                )
            
            try:
                self._tokenizer = run_with_timeout(_load_main_tokenizer, 60)  # 60 second timeout
                print("Main tokenizer loaded")
            except (TimeoutException, Exception) as e:
                raise RuntimeError(f"Failed to load tokenizer after timeout: {e}")
        return self._tokenizer

    @property 
    def trainer(self):
        """Lazy loading of trainer with timeout"""
        if self._trainer is None:
            def _initialize_trainer_wrapper():
                print("Initializing trainer (this may take a few minutes)...")
                self._initialize_trainer()
                return True
            
            try:
                run_with_timeout(_initialize_trainer_wrapper, 300)  # 5 minute timeout for trainer initialization
                print("Trainer initialized")
            except (TimeoutException, Exception) as e:
                raise RuntimeError(f"Failed to initialize trainer after timeout: {e}")
        return self._trainer
        
    def _initialize_trainer(self):
        """Initialize trainer with optimizations and safety checks"""
        tokenizer = self.tokenizer
        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
                "requirement"
            )

        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        self.max_seq_length = max_seq_length

        # Garbage collection before heavy operations
        gc.collect()
        
        # Cache entity type knowledge with timeout
        if self._entity_type_cache is None:
            print("Loading entity type knowledge...")
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
                print(f"Loaded {len(entity_type_knowledge)} entity types")
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

        print("Tokenizing entity type descriptions...")
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
        
        # Initialize model with optimizations and safety checks
        print("Loading Binder model...")
        print(f"Binder model path: {model_args.binder_model_name_or_path}")
        print(f"Base model path: {model_args.model_name_or_path}")
        
        # Load the config from the repository first, then override with local settings
        try:
            # Try to load the config from the repository
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
            print(f"Created BinderConfig with repository settings")
            
        except Exception as config_error:
            print(f"Could not load repository config: {config_error}")
            print("Falling back to local config creation")
            
            # Fallback to manual config creation
            config = BinderConfig(
                pretrained_model_name_or_path=model_args.model_name_or_path,  # Use base model for config
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
        
        # Load model with comprehensive workaround for known issues
        print(f"Loading Binder model from: {model_args.binder_model_name_or_path}")
        
        def load_model_with_workarounds():
            """Comprehensive model loading with multiple fallback strategies"""
            
            # Strategy 1: Direct loading (original approach)
            try:
                print("Strategy 1: Direct model loading...")
                model = Binder.from_pretrained(model_args.binder_model_name_or_path, config=config)
                print("Strategy 1 successful: Model loaded directly")
                return model
            except Exception as e1:
                print(f"Strategy 1 failed: {e1}")
            
            # Strategy 2: Load with trust_remote_code (for model_type issues)
            try:
                print("Strategy 2: Loading with trust_remote_code...")
                model = Binder.from_pretrained(
                    model_args.binder_model_name_or_path,
                    config=config,
                    trust_remote_code=True
                )
                print("Strategy 2 successful: Model loaded with trust_remote_code")
                return model
            except Exception as e2:
                print(f"Strategy 2 failed: {e2}")
            
            # Strategy 3: Force download to avoid cache issues
            try:
                print("Strategy 3: Force download to avoid cache corruption...")
                model = Binder.from_pretrained(
                    model_args.binder_model_name_or_path,
                    config=config,
                    force_download=True,
                    resume_download=True
                )
                print("Strategy 3 successful: Model loaded with force download")
                return model
            except Exception as e3:
                print(f"Strategy 3 failed: {e3}")
            
            # If all strategies fail, raise error to identify the real issue
            print("All direct loading strategies failed")
            raise RuntimeError(f"Cannot load model from {model_args.binder_model_name_or_path}. All strategies failed. Please fix the model repository.")
        
        # Execute model loading with comprehensive error handling
        try:
            model = run_with_timeout(load_model_with_workarounds, 180)  # 3 minute timeout
            print("Model loading completed successfully")
        except TimeoutException:
            print("Model loading timed out - creating fallback model")
            model = Binder(config)
        except Exception as final_error:
            print(f"All model loading strategies failed: {final_error}")
            print("Creating minimal fallback model")
            raise RuntimeError
        
        # Verify model is working
        try:
            model.eval()
            print("Model set to evaluation mode")
        except Exception as eval_error:
            print(f"Could not set model to eval mode: {eval_error}")
        
        # Move to correct device with memory check
        if self.device != "cpu" and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).total_memory < 2 * 1024**3:  # Less than 2GB VRAM
                print("⚠️  Limited GPU memory detected. Consider using CPU mode.")
            model = model.cuda()
            print("Model moved to GPU")
        else:
            print("Using CPU for inference")

        # Data collator
        data_collator = BinderDataCollator(
            type_input_ids=tokenized_descriptions["input_ids"],
            type_attention_mask=tokenized_descriptions["attention_mask"],
            type_token_type_ids=tokenized_descriptions["token_type_ids"] if "token_type_ids" in tokenized_descriptions else None,
        )

        # Post-processing:
        def post_processing_function(examples, features, predictions, stage=f"eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            # Apply threshold factor modification
            if self.prediction_threshold_factor != 1.0:
                # Modify the span logits by applying threshold factor
                if len(predictions) == 4:
                    input_ids, start_logits, end_logits, span_logits = predictions
                    # Apply threshold factor by scaling CLS logits comparison
                    modified_predictions = (input_ids, start_logits, end_logits, span_logits)
                    
                    # Import the custom postprocessing function
                    from src.utils import postprocess_nested_predictions_with_threshold
                    
                    # Try to use custom function with threshold factor
                    try:
                        metrics = postprocess_nested_predictions_with_threshold(
                            examples=examples,
                            features=features,
                            predictions=modified_predictions,
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
                # Use original postprocessing
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

        # Initialize our Trainer
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
        """Optimized text tokenization with caching and timeout protection"""
        text_hash = hash(text)
        if text_hash in self._cached_features:
            return self._cached_features[text_hash]
        
        def _tokenize_with_timeout():
            # Use faster approach for sentence tokenization
            sentence_spans = list(self.ru_tokenizer.span_tokenize(text))
            
            offset_mapping = []
            for span in sentence_spans:
                start, end = span
                context = text[start:end]
                word_spans = list(self.word_tokenizer.span_tokenize(context))
                offset_mapping.extend([(s + start, e + start) for s, e in word_spans])
            
            return offset_mapping
            
        try:
            offset_mapping = run_with_timeout(_tokenize_with_timeout, 30)  # 30 second timeout
        except (TimeoutException, Exception) as e:
            print(f"Tokenization timeout or error: {e}. Using simple fallback.")
            # Simple fallback tokenization
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
        
        # Cache result if reasonable size
        if len(self._cached_features) < 1000:  # Prevent memory bloat
            self._cached_features[text_hash] = result
            
        return result

    def predict_batch(self, texts):
        """Optimized batch prediction for multiple texts"""
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        
        if len(texts) == 1:
            return self._predict_single(texts[0])
        
        # True batch processing for multiple texts
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
            
        # Force consistent padding for batch processing
        original_pad_setting = data_args.pad_to_max_length
        data_args.pad_to_max_length = True
            
        try:
            # Predict Feature Creation
            with training_args.main_process_first(desc="batch prediction dataset map pre-processing"):
                predict_dataset = self.map(
                    predict_examples,
                    lambda x: self._prepare_validation_features(x, "test"),
                    remove_columns=predict_examples.column_names,
                )

            # Disable gradient computation for inference
            with torch.no_grad():
                results = trainer.predict(
                    predict_dataset, 
                    predict_examples, 
                    ignore_keys=["offset_mapping", "example_id", "split", "token_start_mask", "token_end_mask"]
                )
        finally:
            # Restore original padding setting
            data_args.pad_to_max_length = original_pad_setting

        # Process predictions for each text
        predictions = results.predictions
        
        # Group predictions by original text ID
        # predictions is a set of Annotation objects, each with an 'id' field
        text_predictions = [[] for _ in texts]
        
        # Create a mapping from example ID to text index
        id_to_text_index = {}
        for i, text in enumerate(texts):
            id_to_text_index[str(i)] = i  # IDs are strings in the dataset
        
        for p in predictions:
            text_id_str = str(p.id)  # Annotation.id is the example ID as string
            if text_id_str in id_to_text_index:
                text_index = id_to_text_index[text_id_str]
                text_predictions[text_index].append((p.start_char, p.end_char, p.entity_type, p.text))
        
        # Sort predictions for each text
        for i in range(len(text_predictions)):
            text_predictions[i] = sorted(text_predictions[i], key=lambda x: (x[0], x[1]))
        
        return text_predictions

    def _predict_single(self, text):
        """Internal method for single text prediction"""
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
            
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = self.map(
                predict_examples,
                lambda x: self._prepare_validation_features(x, "test"),
                remove_columns=predict_examples.column_names,
            )

        # Disable gradient computation for inference
        with torch.no_grad():
            results = trainer.predict(
                predict_dataset, 
                predict_examples, 
                ignore_keys=["offset_mapping", "example_id", "split", "token_start_mask", "token_end_mask"]
            )

        # Process predictions
        predictions = results.predictions
        processed_predictions = []
        for p in predictions:
            processed_predictions.append((p.start_char, p.end_char, p.entity_type, p.text))
        processed_predictions = sorted(processed_predictions, key=lambda x: (x[0], x[1]))
        
        return processed_predictions

    def predict(self, text):
        """Optimized single text prediction"""
        return self._predict_single(text)
        
    def warm_up(self, sample_text="Это тестовый текст для прогрева модели."):
        """Warm up the model with a sample prediction to optimize subsequent calls"""
        _ = self.predict(sample_text)
        
    def clear_cache(self):
        """Clear feature cache to free memory"""
        self._cached_features.clear()
        
    def enable_batch_mode(self):
        """Enable optimizations for batch processing with memory awareness"""
        self._batch_mode = True
        
        # Check available memory before increasing batch size
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        if available_memory < self.max_memory_gb:
            # Conservative batch size for limited memory
            target_batch_size = min(4, self.training_args.per_device_eval_batch_size)
            print(f"Limited memory ({available_memory:.1f}GB < {self.max_memory_gb}GB). Using conservative batch size: {target_batch_size}")
        else:
            # Moderate increase in batch size for better GPU utilization, but respect memory limits
            target_batch_size = min(16, self.training_args.per_device_eval_batch_size * 2)
            print(f"Sufficient memory. Using batch size: {target_batch_size}")
            
        self.training_args.per_device_eval_batch_size = target_batch_size

    def map(self, dataset, function, remove_columns):
        """
        Simplified map function that delegates to the dataset's built-in map method.
        This avoids compatibility issues with different versions of the datasets library.
        """
        return dataset.map(
            function,
            batched=True,
            remove_columns=remove_columns,
            desc="Processing dataset features"
        )

    def _prepare_validation_features(self, examples, split):

        # print(examples)

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

        # For evaluation, we will need to convert our predictions to spans of the text, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["split"] = []
        tokenized_examples["example_id"] = []
        tokenized_examples["token_start_mask"] = []
        tokenized_examples["token_end_mask"] = []

        # Grab the sequence corresponding to that example (to know what is the text and what are special tokens).
        for i in range(len(tokenized_examples["input_ids"])):
            tokenized_examples["split"].append(split)

            # Grab the sequence corresponding to that example (to know what is the text and what are special tokens).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several texts, this is the index of the example containing this text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Create token_start_mask and token_end_mask where mask = 1 if the corresponding token is either a start
            # or an end of a word in the original dataset.
            token_start_mask, token_end_mask = [], []
            word_start_chars = examples["word_start_chars"][sample_index]
            word_end_chars = examples["word_end_chars"][sample_index]
            for index, (start_char, end_char) in enumerate(tokenized_examples["offset_mapping"][i]):
                if sequence_ids[index] != 0:
                    token_start_mask.append(0)
                    token_end_mask.append(0)
                else:
                    # print(word_start_chars)
                    # print(start_char)
                    token_start_mask.append(int(start_char in word_start_chars))
                    token_end_mask.append(int(end_char in word_end_chars))

            tokenized_examples["token_start_mask"].append(token_start_mask)
            tokenized_examples["token_end_mask"].append(token_end_mask)

            # Set to None the offset_mapping that are not part of the text so it's easy to determine if a token
            # position is part of the text or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 0 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

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
    """
    print("="*80)
    print("RESOURCE-CONSTRAINED STREAMING BATCH PREDICTION DEMO")
    print("="*80)
    
    # Set process priority to prevent system freeze
    try:
        p = psutil.Process()
        if platform.system() == "Windows":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:  # Linux/Mac
            p.nice(10)  # Lower priority
        print("Set process priority to low to prevent system freeze")
    except Exception as e:
        print(f"Could not set process priority: {e}")
    
    # Configuration with STRICT resource limits
    inference_config_path = "./inference/inference-config-optimized.json"
    dataset_path = "./data/seccol_events_texts_1500_new2-div/test"
    predicted_dataset_path = "./data/seccol_events_texts_1500_new2-div/test_predicted/"
    
    # CONSERVATIVE batch processing parameters to prevent freezing
    BATCH_SIZE = 3  # Very conservative batch size
    MEMORY_LIMIT_GB = 16  # Memory limit for processing
    MAX_FILES = 1000 # Limit total files to prevent system overload
    TIMEOUT_PER_BATCH = 120  # 2 minute timeout per batch
    INIT_TIMEOUT = 300  # 5 minute timeout for initialization
    
    # Memory check before starting
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    print(f"Memory status: {available_gb:.1f}GB available, {mem.total / 1024**3:.1f}GB total")
    
    if available_gb < 4:
        BATCH_SIZE = 2
        MAX_FILES = 10
        print(f"Very low memory! Reducing limits: batch_size={BATCH_SIZE}, max_files={MAX_FILES}")
    elif available_gb < 8:
        BATCH_SIZE = 2
        MAX_FILES = 15
        print(f"Low memory! Reducing limits: batch_size={BATCH_SIZE}, max_files={MAX_FILES}")
    
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {predicted_dataset_path}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Memory limit: {MEMORY_LIMIT_GB}GB")
    print(f"Max files: {MAX_FILES}")
    print(f"Timeout per batch: {TIMEOUT_PER_BATCH}s")
    
    # Initialize inference engine with timeouts and error handling
    print(f"\nInitializing BinderInference (timeout: {INIT_TIMEOUT}s)...")
    start_time = time.time()
    
    try:
        inf = run_with_timeout(
            lambda: BinderInference(
                config_path=inference_config_path,
                max_memory_gb=MEMORY_LIMIT_GB,
                device="auto"
            ), 
            INIT_TIMEOUT
        )
        print("BinderInference initialized successfully")
    except TimeoutException:
        print("Initialization timed out! This prevents system freeze.")
        return {}
    except Exception as e:
        print(f"Initialization failed: {e}")
        return {}
    
    # Monitor memory usage after initialization
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory after init: {memory_usage:.1f}MB")
    
    # Warm up the model with timeout protection
    print("Warming up model...")
    try:
        run_with_timeout(
            lambda: inf.warm_up("Это тестовый текст для российских кибербезопасности."),
            30  # 30 second timeout for warmup
        )
        print("Model warmup complete")
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")
    
    init_time = time.time() - start_time
    print(f"Initialization complete: {init_time:.2f}s")
    
    # Collect text files with limits to prevent system overload
    print(f"\nCollecting text files (max {MAX_FILES})...")
    text_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for f in sorted(files):
            if f.endswith(".txt") and len(text_files) < MAX_FILES:
                text_files.append(os.path.join(root, f))
            if len(text_files) >= MAX_FILES:
                break
        if len(text_files) >= MAX_FILES:
            break
    
    print(f"Found {len(text_files)} text files (limited to {MAX_FILES})")
    
    if not text_files:
        print("No text files found!")
        return {}
    
    # Create output directory
    os.makedirs(predicted_dataset_path, exist_ok=True)
    
    # Batch processing with strict resource monitoring
    print(f"\nStarting resource-constrained batch processing...")
    total_start_time = time.time()
    
    text2pred = {}
    total_entities = 0
    total_chars = 0
    processed_files = 0
    failed_files = 0
    
    # Process files in batches with comprehensive monitoring
    for batch_start in tqdm(range(0, len(text_files), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(text_files))
        batch_files = text_files[batch_start:batch_end]
        
        # Memory monitoring before each batch
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        available_mem = psutil.virtual_memory().available / (1024**3)  # GB
        
        if available_mem < 2:  # Less than 2GB available
            print(f"Critical memory shortage ({available_mem:.1f}GB). Stopping to prevent freeze.")
            break
        
        if current_memory > 4000:  # More than 4GB process memory
            print(f"High process memory usage ({current_memory:.1f}MB). Running garbage collection.")
            gc.collect()
            time.sleep(1)  # Brief pause for cleanup
        
        # Load batch texts with size limits
        batch_texts = []
        batch_filenames = []
        batch_file_mapping = {}
        
        for file_path in batch_files:
            try:
                with open(file_path, "r", encoding="UTF-8") as tf:
                    text = tf.read().strip()
                    # Skip empty files and very large files to prevent memory issues
                    if text and len(text) < 50000:  # Skip files larger than 50KB
                        batch_texts.append(text)
                        filename = os.path.basename(file_path)
                        batch_filenames.append(filename)
                        batch_file_mapping[len(batch_texts) - 1] = {
                            'filename': filename,
                            'text': text,
                            'file_path': file_path
                        }
                        total_chars += len(text)
                    elif len(text) >= 50000:
                        print(f"Skipping large file {os.path.basename(file_path)} ({len(text)} chars)")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                failed_files += 1
                continue
        
        if not batch_texts:
            continue
        
        # Batch prediction with timeout protection
        batch_start_time = time.time()
        try:
            batch_predictions = run_with_timeout(
                lambda: inf.predict_batch(batch_texts),
                TIMEOUT_PER_BATCH
            )
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
            
        except TimeoutException:
            print(f"  Batch prediction timed out after {TIMEOUT_PER_BATCH}s")
            failed_files += len(batch_texts)
            # Skip fallback processing to prevent further timeouts
            
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            failed_files += len(batch_texts)
            # Conservative fallback - only try individual processing if batch was small
            if len(batch_texts) <= 2:
                print(f"  Attempting individual fallback for {len(batch_texts)} files...")
                for i, text in enumerate(batch_texts):
                    try:
                        pred = run_with_timeout(
                            lambda: inf.predict(text),
                            30  # 30 second timeout per individual prediction
                        )
                        filename = batch_filenames[i]
                        text2pred[filename] = {'text': text, 'preds': pred}
                        total_entities += len(pred)
                        processed_files += 1
                    except Exception as e2:
                        print(f" Individual prediction failed for {batch_filenames[i]}: {e2}")
                        failed_files += 1
            else:
                print(f"  ⚠ Skipping fallback for large batch to prevent further issues")
        
        # Aggressive memory management to prevent accumulation
        if batch_start % (BATCH_SIZE * 5) == 0:  # Every 5 batches instead of 10
            print(" Running memory cleanup...")
            if hasattr(inf, 'clear_cache'):
                inf.clear_cache()
            gc.collect()
            time.sleep(0.5)  # Brief pause for cleanup
    
    total_processing_time = time.time() - total_start_time
    
    print(f"\n RESOURCE-CONSTRAINED PROCESSING SUMMARY:")
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
    
    print(f"\nProcessing completed safely without system freeze!")
    
    return text2pred


if __name__ == "__main__":
    # Run the efficient streaming batch prediction demo
    text2pred = stream_batch_prediction_demo()
    
    # Original evaluation logic (kept for compatibility)
    print(f"\n Loading gold annotations...")
    
    dataset_path = "./data/seccol_events_texts_1500_new2-div/test"
    predicted_dataset_path = "./data/seccol_events_texts_1500_new2-div/test_predicted/"
    
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
        metrics[tag] = {"tp" : 0, "fp" : 0, "fn" : 0}
    metrics["total"] = {"tp" : 0, "fp" : 0, "fn" : 0}
    for file in text2pred.keys():
        # print(text2pred[file])
        # print(file)
        try:
            golds = set(text2pred[file]["golds"])
            preds = set(text2pred[file]["preds"])
        except KeyError:
            print(file)
            continue

        # print(sorted(list(golds)))
        # print(sorted(list(preds)))

        with open(predicted_dataset_path + file, "w", encoding = "UTF-8") as tf:
            tf.write(text2pred[file]["text"])

        with open(predicted_dataset_path + file.replace('.txt','.ann'), "w", encoding = "UTF-8") as af:
            for p_idx, p in enumerate(sorted(list(preds))):
                s, e, t, sp = p
                af.write("T" + str(p_idx + 1) + "\t" + t + " " + str(s) + " " + str(e) + "\t" + sp + "\n")

        ts = compute_tp_fn_fp(preds, golds)
        tp, fn, fp = ts["tp"], ts["fn"], ts["fp"]
        metrics["total"]["tp"] += tp
        metrics["total"]["fp"] += fp
        metrics["total"]["fn"] += fn

        for tag in tags:
            tagged_golds = set([g for g in golds if g[2] == tag])
            tagged_preds = set([p for p in preds if p[2] == tag])
            tagged_ts = compute_tp_fn_fp(tagged_preds, tagged_golds)
            tagged_tp, tagged_fp, tagged_fn = tagged_ts["tp"], tagged_ts["fn"], tagged_ts["fp"]
            metrics[tag]["tp"] += tagged_tp
            metrics[tag]["fp"] += tagged_fp
            metrics[tag]["fn"] += tagged_fn

    metrics["total"] = {**metrics["total"], **compute_precision_recall_f1(metrics["total"]["tp"], metrics["total"]["fn"], metrics["total"]["fp"])}
    for tag in tags:
        metrics[tag] = {**metrics[tag], **compute_precision_recall_f1(metrics[tag]["tp"], metrics[tag]["fn"], metrics[tag]["fp"])}

    print(metrics)
        


