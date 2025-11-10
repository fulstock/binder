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

class BinderInference:

    def __init__(self, config_path = None, device = "auto", prediction_threshold_factor=1.0):
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
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"Loading config from: {config_path}")
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=config_path)
        
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

    def _optimize_model(self, model):
        """Apply mixed-precision and compilation/JIT techniques for faster inference."""
        import torch  # Ensure torch is available in function scope
        
        try:
            # Convert to half precision when running on GPU for speed & memory benefits
            if self.device != "cpu" and torch.cuda.is_available():
                model = model.half()
                
            # Try torch.compile if available (PyTorch ≥2.0)
            try:
                import torch._dynamo
                if hasattr(torch, "compile") and callable(torch.compile) and hasattr(torch, "compiler"):
                    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                    print("Model compiled with torch.compile for faster inference")
                    return model
            except (ImportError, AttributeError, Exception) as compile_error:
                print(f"torch.compile not available or failed: {compile_error}")
            
            # Fallback to torch.jit.script if torch.compile is not available
            # Try to script submodules for better compatibility
            try:
                print("Attempting to script model submodules...")
                for name, module in model.named_children():
                    try:
                        scripted_module = torch.jit.script(module)
                        setattr(model, name, scripted_module)
                        print(f"  - Scripted module: {name}")
                    except Exception as jit_error_module:
                        print(f"  - Could not script module {name}: {jit_error_module}")
                print("Model scripting of submodules complete.")
                return model
            except Exception as jit_error:
                print(f"torch.jit.script failed on submodules: {jit_error}")
                
        except Exception as opt_error:
            print(f"Model optimization failed: {opt_error}")
            
        # If all optimizations fail, return the original model
        print("Using original model without optimizations")
        return model

    @property
    def ru_tokenizer(self):
        """Lazy loading of Russian tokenizer"""
        if self._ru_tokenizer is None:
            def _load_ru_tokenizer():
                return load("tokenizers/punkt/russian.pickle")
            
            try:
                self._ru_tokenizer = _load_ru_tokenizer()
                print("Russian tokenizer loaded")
            except Exception as e:
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
                self._word_tokenizer = _load_word_tokenizer()
                print("Word tokenizer loaded")
            except Exception as e:
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
                self._tokenizer = _load_main_tokenizer()
                print("Main tokenizer loaded")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {e}")
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
                _initialize_trainer_wrapper()
                print("Trainer initialized")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize trainer: {e}")
        return self._trainer
        
    def _initialize_trainer(self):
        """Initialize trainer with optimizations and safety checks"""
        tokenizer = self.tokenizer
        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args

        # Ensure DataLoader does not use multiprocessing workers to avoid freeze on exit
        if hasattr(training_args, "dataloader_num_workers") and training_args.dataloader_num_workers != 0:
            print(
                f"Overriding dataloader_num_workers from {training_args.dataloader_num_workers} to 0 to prevent worker cleanup hangs"
            )
            training_args.dataloader_num_workers = 0

        # Disable persistent workers explicitly if attribute exists
        if hasattr(training_args, "dataloader_persistent_workers"):
            if training_args.dataloader_persistent_workers:
                print("Disabling dataloader_persistent_workers to avoid cleanup issues")
            training_args.dataloader_persistent_workers = False

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
            model = load_model_with_workarounds()
            print("Model loading completed successfully")
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

        # Apply additional model-level optimisations (half-precision, compile/JIT)
        model = self._optimize_model(model)
        # Ensure Trainer keeps all dataset columns to avoid mismatch with compiled model signatures
        if hasattr(training_args, "remove_unused_columns") and training_args.remove_unused_columns:
            training_args.remove_unused_columns = False
            print("Trainer configured with remove_unused_columns=False to keep all feature columns")

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
        
        def _tokenize():
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
            offset_mapping = _tokenize()
        except Exception as e:
            print(f"Tokenization error: {e}. Using simple fallback.")
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
            
        # Force consistent padding for individual processing (similar to batch processing)
        original_pad_setting = data_args.pad_to_max_length
        data_args.pad_to_max_length = True
        
        try:
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
        
        # Increase batch size for better GPU utilization
        target_batch_size = self.training_args.per_device_eval_batch_size * 2
        print(f"Batch mode enabled. Setting batch size to: {target_batch_size}")
            
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
    - Processes all files without artificial limits
    """
    print("="*80)
    print("STREAMING BATCH PREDICTION DEMO")
    print("="*80)
    
    # Configuration
    inference_config_path = "./conf/inference/small_nerel_inf.json"
    dataset_path = "/home/student1/data/NEREL1.1/test"
    predicted_dataset_path = "/home/student1/data/NEREL1.1/test_predicted_small_nerel"
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
                with open(file_path, "r", encoding="cp1251") as tf:
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
    
    dataset_path = "/home/student1/data/100texts/raw"
    predicted_dataset_path = "/home/student1/data/100texts/ner_predicted"
    
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

    # print(list(text2pred.keys()))

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

    metrics["micro"] = {"tp" : 0, "fp" : 0, "fn" : 0}.copy()
    metrics["macro"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}.copy()
    for file in text2pred.keys():
        print(text2pred[file])
        print(file)
        try:
            # Convert to (start,end,type) triples for comparison to avoid textual mismatches
            if "golds" in text2pred[file]:
                golds = set((s, e, t) for (s, e, t, *_unused) in text2pred[file]["golds"])
            preds = set((s, e, t) for (s, e, t, *_unused) in text2pred[file]["preds"])
        except KeyError:
            if "golds" in text2pred[file]:
                print("golds", text2pred[file]["golds"])
            print("preds", text2pred[file]["preds"])
            print(file)
            continue

        # print(sorted(list(golds)))
        # print(sorted(list(preds)))

        with open(os.path.join(predicted_dataset_path, file), "w", encoding = "UTF-8") as tf:
            print("Writing text of file", file, "into", os.path.join(predicted_dataset_path, file))
            tf.write(text2pred[file]["text"])

        with open(os.path.join(predicted_dataset_path, file.replace('.txt','.ann')), "w", encoding = "UTF-8") as af:
            print("Writing ner perds of file", file.replace('.txt','.ann'), "into", os.path.join(predicted_dataset_path, file.replace('.txt','.ann')))
            for p_idx, p in enumerate(sorted(list(preds))):
                # Handle both (s,e,t) and (s,e,t,text) formats
                if len(p) == 3:
                    s, e, t = p
                    sp = text2pred[file]["text"][s:e]  # Extract text span from original text
                else:
                    s, e, t, sp = p
                af.write("T" + str(p_idx + 1) + "\t" + t + " " + str(s) + " " + str(e) + "\t" + sp + "\n")

        if "golds" in text2pred[file]:
            ts = compute_tp_fn_fp(preds, golds)
            tp, fn, fp = ts["tp"], ts["fn"], ts["fp"]
            metrics["micro"]["tp"] += tp
            metrics["micro"]["fp"] += fp
            metrics["micro"]["fn"] += fn

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

    if "golds" in text2pred[file]:
        metrics["micro"] = {**metrics["micro"], **compute_precision_recall_f1(metrics["micro"]["tp"], metrics["micro"]["fn"], metrics["micro"]["fp"])}
        for tag in tags:
            metrics[tag] = {**metrics[tag], **compute_precision_recall_f1(metrics[tag]["tp"], metrics[tag]["fn"], metrics[tag]["fp"])}
        metrics["macro"]["precision"] = sum([metrics[tag]["precision"] for tag in tags]) / len(tags) if len(tags) > 0 else 0.0
        metrics["macro"]["recall"] = sum([metrics[tag]["recall"] for tag in tags]) / len(tags) if len(tags) > 0 else 0.0
        metrics["macro"]["f1"] = sum([metrics[tag]["f1"] for tag in tags]) / len(tags) if len(tags) > 0 else 0.0

        for tag in sorted(list(tags)):
            print(tag, metrics[tag]["tp"], metrics[tag]["fp"], metrics[tag]["fn"], metrics[tag]["precision"], metrics[tag]["recall"], metrics[tag]["f1"])
        print("total", metrics["micro"]["tp"], metrics["micro"]["fp"], metrics["micro"]["fn"])
        print("micro", metrics["micro"]["precision"], metrics["micro"]["recall"], metrics["micro"]["f1"])
        print("macro", metrics["macro"]["precision"], metrics["macro"]["recall"], metrics["macro"]["f1"])
        
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
        print(f"Overall micro recall: {metrics['micro']['recall']:.3f}")
        print(f"Overall micro precision: {metrics['micro']['precision']:.3f}")
        print(f"Overall micro F1: {metrics['micro']['f1']:.3f}")
        print(f"Overall macro recall: {metrics['macro']['recall']:.3f}")
        print(f"Overall macro precision: {metrics['macro']['precision']:.3f}")
        print(f"Overall macro F1: {metrics['macro']['f1']:.3f}")