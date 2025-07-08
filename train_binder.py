#!/usr/bin/env python3
"""
This script encapsulates the Binder model training process into a reusable class.
It is designed to be imported and used in other modules, providing a clean
interface for training Binder models.

The BinderTraining class provides several usage patterns:

1. Training only (fastest option):
   ```python
   trainer = BinderTraining("conf/lexical-conf.json")
   model = trainer.train_only()
   ```

2. Training with evaluation:
   ```python
   trainer = BinderTraining("conf/lexical-conf.json")
   model = trainer.train_and_evaluate()
   ```

3. Full training, evaluation, and prediction:
   ```python
   trainer = BinderTraining("conf/lexical-conf.json")
   model = trainer.train_evaluate_and_predict()
   ```

4. Flexible training with custom options:
   ```python
   trainer = BinderTraining("conf/lexical-conf.json")
   model = trainer.train(do_eval=True, do_predict=False)
   ```

5. Evaluation or prediction only (requires pre-trained model):
   ```python
   trainer = BinderTraining("conf/lexical-conf.json")
   # ... load or train model first ...
   eval_results = trainer.evaluate_only()
   pred_results = trainer.predict_only()
   ```

6. Safetensors model saving and loading for inference:
   ```python
   # After training
   trainer = BinderTraining("conf/lexical-conf.json")
   trainer.train_only()
   trainer.save_model_as_safetensors("./my_model")  # Manual save
   
   # For inference in another module
   inference_trainer = BinderTraining("conf/lexical-conf.json")
   model = inference_trainer.load_model_for_inference("./my_model")
   ```

The main focus is on training, with validation and testing as optional components.
This makes the training process more flexible and allows for training-only workflows
when evaluation and prediction are not needed.

Models are automatically saved in safetensors format after training for efficient
inference use. The safetensors format provides better security, faster loading,
and cross-platform compatibility compared to pickle-based formats.
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config import BinderConfig
from src.model import Binder
from src.trainer import BinderDataCollator, BinderTrainer
from src import utils as postprocess_utils

# Use the same argument classes as run_ner.py
from run_ner import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


class BinderTraining:
    """A class to encapsulate the Binder model training pipeline."""

    def __init__(self, config_path: str, device: str = "auto"):
        """
        Initializes the training process with a configuration file.

        Args:
            config_path (str): Path to the JSON configuration file for training.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        self.config_path = config_path
        # Store desired device ('cpu', 'cuda', or 'auto') so setup() can adjust TrainingArguments accordingly
        self.device = device
        self.model_args = None
        self.data_args = None
        self.training_args = None
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        
        # Training components (required)
        self.train_dataset = None
        
        # Validation components (optional)
        self.eval_dataset = None
        self.eval_examples = None
        
        # Testing components (optional)
        self.predict_dataset = None
        self.predict_examples = None
        
        # Shared components
        self.entity_type_id_to_str = None
        self.entity_type_str_to_id = None
        self.max_seq_length = None
        
    def setup(self):
        """Parses arguments and sets up logging and environment variables."""
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        self.model_args, self.data_args, self.training_args = parser.parse_json_file(
            json_file=os.path.abspath(self.config_path)
        )

        # ------------------------------------------------------------------
        # Device handling (mirrors BinderInference/text2ner logic)
        # ------------------------------------------------------------------
        if self.device == "cpu":
            # Custom flag (used elsewhere in the code-base) to force CPU execution
            self.training_args.use_cpu = True  # type: ignore[attr-defined]
            # Also align HuggingFace flag to disable CUDA completely
            self.training_args.no_cuda = True  # type: ignore[attr-defined]
        elif self.device == "cuda":
            # Ensure CUDA is enabled if available
            self.training_args.use_cpu = False  # type: ignore[attr-defined]
            self.training_args.no_cuda = False  # type: ignore[attr-defined]
        # If 'auto', leave default behavior untouched (framework decides)

        # Setup environment variables
        os.environ["WANDB_PROJECT"] = self.data_args.wandb_project or self.data_args.dataset_name
        os.environ["WANDB_DIR"] = self.training_args.output_dir

        # Setup logging
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        log_file_handler = logging.FileHandler(os.path.join(self.training_args.output_dir, "run.log"), "a")
        # `transformers` may configure the root logger before we get here which prevents `basicConfig` from
        # adding our handlers.  Using `force=True` (Python ≥3.8) overrides any existing configuration so that
        # our stream & file handlers are correctly attached and log messages become visible.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout), log_file_handler],
            force=True,
        )

        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        transformers.utils.logging.add_handler(log_file_handler)

        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}, "
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {self.training_args}")

        set_seed(self.training_args.seed)

    def _setup_tokenizer_and_entity_types(self):
        """Sets up tokenizer and entity type knowledge (shared by all dataset types)."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
            add_prefix_space=True,
        )
        
        # Set max sequence length
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

        # Load entity type knowledge
        entity_type_knowledge = load_dataset(
            "json", data_files=self.data_args.entity_type_file, cache_dir=self.model_args.cache_dir
        )["train"]
        entity_type_knowledge = entity_type_knowledge.filter(
            lambda example: (
                example["dataset"] == self.data_args.dataset_name and (
                    len(self.data_args.dataset_entity_types) == 0 or
                    example[self.data_args.entity_type_key_field] in self.data_args.dataset_entity_types
                )
            )
        )
        entity_type_knowledge = entity_type_knowledge.sort(self.data_args.entity_type_key_field)

        self.entity_type_id_to_str = [et[self.data_args.entity_type_key_field] for et in entity_type_knowledge]
        self.entity_type_str_to_id = {t: i for i, t in enumerate(self.entity_type_id_to_str)}

        def prepare_type_features(examples):
            tokenized_examples = self.tokenizer(
                examples[self.data_args.entity_type_desc_field],
                truncation=True,
                max_length=self.max_seq_length,
                padding="longest" if len(entity_type_knowledge) <= 1000 else "max_length",
            )
            return tokenized_examples

        with self.training_args.main_process_first(desc="Tokenizing entity type descriptions"):
            tokenized_descriptions = entity_type_knowledge.map(
                prepare_type_features,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on type descriptions",
                remove_columns=entity_type_knowledge.column_names,
            )

        # Set up data collator
        self.data_collator = BinderDataCollator(
            type_input_ids=tokenized_descriptions["input_ids"],
            type_attention_mask=tokenized_descriptions["attention_mask"],
            type_token_type_ids=tokenized_descriptions["token_type_ids"] if "token_type_ids" in tokenized_descriptions else None,
        )

    def prepare_train_dataset(self):
        """Prepares the training dataset (required for training)."""
        if self.data_args.train_file is None:
            raise ValueError("Training requires a train_file to be specified")
        
        # Check if training file exists
        if not os.path.exists(self.data_args.train_file):
            raise FileNotFoundError(f"Training file not found: {self.data_args.train_file}")
        
        logger.info(f"Loading training data from: {self.data_args.train_file}")
        
        # Load training data
        train_data = load_dataset(
            "json", 
            data_files={"train": self.data_args.train_file}, 
            cache_dir=self.model_args.cache_dir
        )["train"]
        
        logger.info(f"Loaded {len(train_data)} raw training examples")
        
        if len(train_data) == 0:
            raise ValueError("Training dataset is empty. Check your training file.")
        
        # Log some statistics about the raw data
        sample_example = train_data[0]
        logger.info(f"Sample training example fields: {list(sample_example.keys())}")
        
        if "entity_types" in sample_example:
            sample_entities = sample_example["entity_types"]
            logger.info(f"Sample example has {len(sample_entities)} entities")
        
        if self.data_args.max_train_samples is not None:
            train_data = train_data.select(range(self.data_args.max_train_samples))
            logger.info(f"Limited training data to {len(train_data)} examples")
        
        # Training preprocessing function
        def prepare_train_features(examples):
            logger.debug(f"Processing batch of {len(examples['text'])} examples")
            
            tokenized_examples = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_seq_length,
                stride=self.data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized_examples.pop("offset_mapping")

            processed_examples = {
                "input_ids": [],
                "attention_mask": [],
                "token_start_mask": [],
                "token_end_mask": [],
                "ner": [],
            }
            if "token_type_ids" in tokenized_examples:
                processed_examples["token_type_ids"] = []

            logger.debug(f"Tokenized into {len(tokenized_examples['input_ids'])} sequences")
            
            valid_examples_count = 0
            skipped_examples_count = 0
            
            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples["input_ids"][i]
                sequence_ids = tokenized_examples.sequence_ids(i)

                # Find text boundaries
                text_start_index = 0
                while sequence_ids[text_start_index] != 0:
                    text_start_index += 1

                text_end_index = len(input_ids) - 1
                while sequence_ids[text_end_index] != 0:
                    text_end_index -= 1

                sample_index = sample_mapping[i]

                # Create token masks
                token_start_mask, token_end_mask = [], []
                word_start_chars = examples["word_start_chars"][sample_index]
                word_end_chars = examples["word_end_chars"][sample_index]
                for index, (start_char, end_char) in enumerate(offsets):
                    if sequence_ids[index] != 0:
                        token_start_mask.append(0)
                        token_end_mask.append(0)
                    else:
                        token_start_mask.append(int(start_char in word_start_chars))
                        token_end_mask.append(int(end_char in word_end_chars))

                default_span_mask = [
                    [
                        (j - i >= 0) * s * e for j, e in enumerate(token_end_mask)
                    ]
                    for i, s in enumerate(token_start_mask)
                ]

                start_negative_mask = [token_start_mask[:] for _ in self.entity_type_id_to_str]
                end_negative_mask = [token_end_mask[:] for _ in self.entity_type_id_to_str]
                span_negative_mask = [[x[:] for x in default_span_mask] for _ in self.entity_type_id_to_str]

                # Process NER annotations
                tokenized_ner_annotations = []
                entity_types = examples["entity_types"][sample_index]
                entity_start_chars = examples["entity_start_chars"][sample_index]
                entity_end_chars = examples["entity_end_chars"][sample_index]
                
                logger.debug(f"Processing sample {sample_index} with {len(entity_types)} entities")
                
                for entity_type, start_char, end_char in zip(entity_types, entity_start_chars, entity_end_chars):
                    if entity_type not in self.entity_type_str_to_id:
                        logger.debug(f"Skipping unknown entity type: {entity_type}")
                        continue
                        
                    if offsets[text_start_index][0] <= start_char and offsets[text_end_index][1] >= end_char:
                        start_token_index, end_token_index = text_start_index, text_end_index
                        
                        while start_token_index <= text_end_index and offsets[start_token_index][0] <= start_char:
                            start_token_index += 1
                        start_token_index -= 1

                        while offsets[end_token_index][1] >= end_char:
                            end_token_index -= 1
                        end_token_index += 1

                        entity_type_id = self.entity_type_str_to_id[entity_type]

                        tokenized_ner_annotations.append({
                            "type_id": entity_type_id,
                            "start": start_token_index,
                            "end": end_token_index,
                        })

                        start_negative_mask[entity_type_id][start_token_index] = 0
                        end_negative_mask[entity_type_id][end_token_index] = 0
                        span_negative_mask[entity_type_id][start_token_index][end_token_index] = 0

                # Debug: Log information about annotation processing
                if len(tokenized_ner_annotations) == 0:
                    logger.debug(f"No valid annotations found for sample {sample_index}. "
                                f"Original entities: {len(entity_types)}, "
                                f"Text length: {len(examples['text'][sample_index])}, "
                                f"Tokenized length: {len(input_ids)}")
                    skipped_examples_count += 1
                    # Instead of skipping, we'll include examples without annotations for negative sampling
                    # This is important for training the model to distinguish between entity and non-entity spans
                else:
                    logger.debug(f"Found {len(tokenized_ner_annotations)} valid annotations for sample {sample_index}")
                    valid_examples_count += 1
                
                # Include all examples (with or without annotations) for proper training
                processed_examples["input_ids"].append(input_ids)
                if "token_type_ids" in tokenized_examples:
                    processed_examples["token_type_ids"].append(tokenized_examples["token_type_ids"][i])
                processed_examples["attention_mask"].append(tokenized_examples["attention_mask"][i])
                processed_examples["token_start_mask"].append(token_start_mask)
                processed_examples["token_end_mask"].append(token_end_mask)

                processed_examples["ner"].append({
                    "annotations": tokenized_ner_annotations,
                    "start_negative_mask": start_negative_mask,
                    "end_negative_mask": end_negative_mask,
                    "span_negative_mask": span_negative_mask,
                    "token_start_mask": token_start_mask,
                    "token_end_mask": token_end_mask,
                    "default_span_mask": default_span_mask,
                })

            logger.debug(f"Batch processed: {valid_examples_count} with annotations, {skipped_examples_count} without annotations")
            return processed_examples

        # Process training dataset
        logger.info("Starting dataset tokenization and preprocessing...")
        with self.training_args.main_process_first(desc="train dataset map pre-processing"):
            self.train_dataset = train_data.map(
                prepare_train_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=train_data.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        
        logger.info("Dataset preprocessing completed!")
        
        # Add detailed logging about the training dataset
        logger.info(f"Training dataset prepared with {len(self.train_dataset)} examples")
        
        # Debug: Check if we have any examples with annotations
        if len(self.train_dataset) > 0:
            logger.info("Analyzing training dataset...")
            sample_example = self.train_dataset[0]
            num_annotations = len(sample_example["ner"]["annotations"])
            logger.info(f"Sample training example has {num_annotations} annotations")
            
            # Count total annotations across first 100 examples for quick sanity check instead of full dataset to avoid slowdown
            quick_check_examples = min(100, len(self.train_dataset))
            total_annotations_quick = sum(len(self.train_dataset[i]["ner"]["annotations"]) for i in range(quick_check_examples))
            logger.info(f"Total annotations across first {quick_check_examples} examples: {total_annotations_quick}")
            
            # Warn if even quick check shows 0 annotations
            if total_annotations_quick == 0:
                logger.warning("⚠️ No annotations found in the first batch of training examples! Check data format.")
        else:
            logger.error("❌ Training dataset is empty after preprocessing!")
            raise ValueError("Training dataset is empty. Check your data file and preprocessing settings.")
        
        logger.info("Training dataset preparation completed successfully!")

    def prepare_validation_dataset(self):
        """Prepares the validation dataset (optional)."""
        if self.data_args.validation_file is None:
            logger.info("No validation file specified, skipping validation dataset preparation")
            return
        
        # Load validation data
        eval_data = load_dataset(
            "json", 
            data_files={"validation": self.data_args.validation_file}, 
            cache_dir=self.model_args.cache_dir
        )["validation"]
        
        if self.data_args.max_eval_samples is not None:
            eval_data = eval_data.select(range(self.data_args.max_eval_samples))
        
        # Validation preprocessing function
        def prepare_validation_features(examples, split: str = "dev"):
            tokenized_examples = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_seq_length,
                stride=self.data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if self.data_args.pad_to_max_length else False,
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

                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == 0 else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

        # Process validation dataset
        with self.training_args.main_process_first(desc="validation dataset map pre-processing"):
            self.eval_dataset = eval_data.map(
                prepare_validation_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=eval_data.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        
        self.eval_examples = eval_data
        logger.info(f"Validation dataset prepared with {len(self.eval_dataset)} examples")

    def prepare_test_dataset(self):
        """Prepares the test dataset (optional)."""
        if self.data_args.test_file is None:
            logger.info("No test file specified, skipping test dataset preparation")
            return
        
        # Load test data
        test_data = load_dataset(
            "json", 
            data_files={"test": self.data_args.test_file}, 
            cache_dir=self.model_args.cache_dir
        )["test"]
        
        if self.data_args.max_predict_samples is not None:
            test_data = test_data.select(range(self.data_args.max_predict_samples))
        
        # Test preprocessing function (same as validation)
        def prepare_test_features(examples):
            tokenized_examples = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_seq_length,
                stride=self.data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            tokenized_examples["split"] = []
            tokenized_examples["example_id"] = []
            tokenized_examples["token_start_mask"] = []
            tokenized_examples["token_end_mask"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                tokenized_examples["split"].append("test")
                sequence_ids = tokenized_examples.sequence_ids(i)
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

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

                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == 0 else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

        # Process test dataset
        with self.training_args.main_process_first(desc="test dataset map pre-processing"):
            self.predict_dataset = test_data.map(
                prepare_test_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=test_data.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
        
        self.predict_examples = test_data
        logger.info(f"Test dataset prepared with {len(self.predict_dataset)} examples")

    def save_model_as_safetensors(self, output_dir: str = None, save_tokenizer: bool = True):
        """
        Saves the trained model in safetensors format for efficient inference.
        
        Args:
            output_dir (str, optional): Directory to save the model. If None, uses training_args.output_dir.
            save_tokenizer (bool): Whether to save the tokenizer alongside the model.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        if output_dir is None:
            output_dir = self.training_args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"💾 Saving model as safetensors to: {output_dir}")
        
        # Save the model with safetensors format
        self.model.save_pretrained(
            output_dir,
            safe_serialization=True,  # This ensures safetensors format
            save_function=None,  # Use default save function
        )
        
        # Save tokenizer if requested
        if save_tokenizer and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info("💾 Tokenizer saved alongside model")
        
        # Save config for inference
        config_dict = {
            "model_type": "binder",
            "entity_types": self.entity_type_id_to_str,
            "max_seq_length": self.max_seq_length,
            "model_args": {
                "hidden_dropout_prob": self.model_args.hidden_dropout_prob,
                "use_span_width_embedding": self.model_args.use_span_width_embedding,
                "linear_size": self.model_args.linear_size,
                "init_temperature": self.model_args.init_temperature,
                "start_loss_weight": self.model_args.start_loss_weight,
                "end_loss_weight": self.model_args.end_loss_weight,
                "span_loss_weight": self.model_args.span_loss_weight,
                "threshold_loss_weight": self.model_args.threshold_loss_weight,
                "ner_loss_weight": self.model_args.ner_loss_weight,
            },
            "data_args": {
                "max_span_length": self.data_args.max_span_length,
                "entity_type_key_field": self.data_args.entity_type_key_field,
                "entity_type_desc_field": self.data_args.entity_type_desc_field,
            }
        }
        
        # Save inference config
        inference_config_path = os.path.join(output_dir, "inference_config.json")
        with open(inference_config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Inference configuration saved to: {inference_config_path}")
        logger.info("✅ Model saved successfully in safetensors format for inference!")
        
        return output_dir

    def load_model_for_inference(self, model_path: str):
        """
        Loads a trained model from safetensors checkpoint for inference.
        
        Args:
            model_path (str): Path to the saved model directory.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        logger.info(f"📂 Loading model from: {model_path}")
        
        # Load inference config
        inference_config_path = os.path.join(model_path, "inference_config.json")
        if os.path.exists(inference_config_path):
            with open(inference_config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            self.entity_type_id_to_str = config_dict.get("entity_types", [])
            self.entity_type_str_to_id = {t: i for i, t in enumerate(self.entity_type_id_to_str)}
            self.max_seq_length = config_dict.get("max_seq_length", 384)
            
            logger.info("📋 Inference configuration loaded")
        else:
            logger.warning("⚠️ No inference_config.json found, using default settings")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("🔤 Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load tokenizer: {e}")
        
        # Load model
        try:
            self.model = Binder.from_pretrained(model_path)
            logger.info("🤖 Model loaded successfully from safetensors")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        logger.info("✅ Model loaded successfully for inference!")
        return self.model

    def train(self, do_eval: bool = None, do_predict: bool = None):
        """
        Trains the Binder model.
        
        Args:
            do_eval (bool, optional): Whether to run evaluation. If None, uses config setting.
            do_predict (bool, optional): Whether to run prediction. If None, uses config setting.
        """
        # Setup and prepare training data (required)
        self.setup()
        self._setup_tokenizer_and_entity_types()
        self.prepare_train_dataset()
        
        # Prepare optional datasets based on parameters or config
        if do_eval is None:
            do_eval = self.training_args.do_eval
        if do_predict is None:
            do_predict = self.training_args.do_predict
            
        if do_eval:
            self.prepare_validation_dataset()
        if do_predict:
            self.prepare_test_dataset()

        # Load model config
        config = BinderConfig(
            pretrained_model_name_or_path=self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
            hidden_dropout_prob=self.model_args.hidden_dropout_prob,
            max_span_width=self.data_args.max_seq_length + 1,
            use_span_width_embedding=self.model_args.use_span_width_embedding,
            linear_size=self.model_args.linear_size,
            init_temperature=self.model_args.init_temperature,
            start_loss_weight=self.model_args.start_loss_weight,
            end_loss_weight=self.model_args.end_loss_weight,
            span_loss_weight=self.model_args.span_loss_weight,
            threshold_loss_weight=self.model_args.threshold_loss_weight,
            ner_loss_weight=self.model_args.ner_loss_weight,
        )

        # Load model
        logger.info(f"Initializing Binder model with {len(self.entity_type_id_to_str)} entity types")
        self.model = Binder(config)
        
        # Validate model initialization
        logger.info(f"Model initialized successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())} total parameters")
        logger.info(f"Model trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
        
        # Set up post-processing function
        def post_processing_function(examples, features, predictions, stage="eval"):
            return postprocess_utils.postprocess_nested_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                id_to_type=self.entity_type_id_to_str,
                max_span_length=self.data_args.max_span_length,
                output_dir=self.training_args.output_dir if self.training_args.should_save else None,
                log_level=self.training_args.get_process_log_level(),
                prefix=stage,
                tokenizer=self.tokenizer,
                train_file=self.data_args.train_file,
            )

        # Initialize Trainer
        # Only use EarlyStoppingCallback when doing evaluation, since it requires load_best_model_at_end=True
        callbacks = []
        if do_eval and self.eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=20))
        
        trainer = BinderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if do_eval else None,
            eval_examples=self.eval_examples if do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=post_processing_function,
            callbacks=callbacks,
            compute_metrics=None,
        )

        # Training (always performed)
        logger.info("*** Starting Training ***")
        checkpoint = None
        # Only resume if user explicitly provided checkpoint path
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        # If user did not provide checkpoint but wants to continue, detect last checkpoint only when overwrite_output_dir is False
        elif not self.training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(self.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Save model as safetensors for inference
        safetensors_dir = os.path.join(self.training_args.output_dir, "safetensors_model")
        self.save_model_as_safetensors(safetensors_dir, save_tokenizer=True)
        
        logger.info("*** Training completed successfully ***")

        # Optional evaluation
        if do_eval and self.eval_dataset is not None:
            logger.info("*** Starting Evaluation ***")
            eval_metrics = trainer.evaluate()
            
            max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
            eval_metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            logger.info("*** Evaluation completed ***")

        # Optional prediction
        if do_predict and self.predict_dataset is not None:
            logger.info("*** Starting Prediction ***")
            results = trainer.predict(self.predict_dataset, self.predict_examples)
            predict_metrics = results.metrics
            
            max_predict_samples = (
                self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(self.predict_dataset)
            )
            predict_metrics["predict_samples"] = min(max_predict_samples, len(self.predict_dataset))

            trainer.log_metrics("predict", predict_metrics)
            trainer.save_metrics("predict", predict_metrics)
            logger.info("*** Prediction completed ***")

        return trainer

    def train_only(self):
        """Trains the model without evaluation or prediction."""
        return self.train(do_eval=False, do_predict=False)

    def train_and_evaluate(self):
        """Trains the model and runs evaluation."""
        return self.train(do_eval=True, do_predict=False)

    def train_evaluate_and_predict(self):
        """Trains the model, runs evaluation, and makes predictions."""
        return self.train(do_eval=True, do_predict=True)

    def evaluate_only(self):
        """Runs evaluation only (requires a trained model)."""
        if self.model is None:
            raise ValueError("Model not initialized. Run training first or load a pre-trained model.")
        
        self.setup()
        self._setup_tokenizer_and_entity_types()
        self.prepare_validation_dataset()
        
        if self.eval_dataset is None:
            raise ValueError("No validation dataset available for evaluation")
        
        # Set up post-processing function
        def post_processing_function(examples, features, predictions, stage="eval"):
            return postprocess_utils.postprocess_nested_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                id_to_type=self.entity_type_id_to_str,
                max_span_length=self.data_args.max_span_length,
                output_dir=self.training_args.output_dir if self.training_args.should_save else None,
                log_level=self.training_args.get_process_log_level(),
                prefix=stage,
                tokenizer=self.tokenizer,
                train_file=self.data_args.train_file,
            )

        # Initialize Trainer for evaluation
        trainer = BinderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=self.eval_dataset,
            eval_examples=self.eval_examples,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=post_processing_function,
            callbacks=[],
            compute_metrics=None,
        )

        logger.info("*** Starting Evaluation ***")
        eval_metrics = trainer.evaluate()
        
        max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
        eval_metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.info("*** Evaluation completed ***")
        
        return trainer

    def predict_only(self):
        """Runs prediction only (requires a trained model)."""
        if self.model is None:
            raise ValueError("Model not initialized. Run training first or load a pre-trained model.")
        
        self.setup()
        self._setup_tokenizer_and_entity_types()
        self.prepare_test_dataset()
        
        if self.predict_dataset is None:
            raise ValueError("No test dataset available for prediction")
        
        # Set up post-processing function
        def post_processing_function(examples, features, predictions, stage="predict"):
            return postprocess_utils.postprocess_nested_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                id_to_type=self.entity_type_id_to_str,
                max_span_length=self.data_args.max_span_length,
                output_dir=self.training_args.output_dir if self.training_args.should_save else None,
                log_level=self.training_args.get_process_log_level(),
                prefix=stage,
                tokenizer=self.tokenizer,
                train_file=self.data_args.train_file,
            )

        # Initialize Trainer for prediction
        trainer = BinderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=None,
            eval_examples=None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=post_processing_function,
            callbacks=[],
            compute_metrics=None,
        )

        logger.info("*** Starting Prediction ***")
        results = trainer.predict(self.predict_dataset, self.predict_examples)
        predict_metrics = results.metrics
        
        max_predict_samples = (
            self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(self.predict_dataset)
        )
        predict_metrics["predict_samples"] = min(max_predict_samples, len(self.predict_dataset))

        trainer.log_metrics("predict", predict_metrics)
        trainer.save_metrics("predict", predict_metrics)
        logger.info("*** Prediction completed ***")
        
        return trainer


if __name__ == "__main__":
    """
    Example of how to run the BinderTraining class.
    This provides a command-line interface for training, similar to run_ner.py.
    """
    print("🚀 Starting Binder Training")
    
    # Default to lexical-conf.json if no other config is provided
    config_file = "conf/train_only.json"
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        config_file = sys.argv[1]
        
    print(f"🔩 Using configuration file: {config_file}")
    
    try:
        binder_trainer = BinderTraining(config_path=config_file)
        
        # Example usage options:
        # 1. Training only (fastest, no evaluation or prediction)
        trainer = binder_trainer.train_only()
        
        # 2. Training with evaluation
        # trainer = binder_trainer.train_and_evaluate()
        
        # 3. Full training, evaluation, and prediction (default behavior)
        # trainer = binder_trainer.train()
        
        # 4. You can also use the original behavior:
        # trainer = binder_trainer.train_evaluate_and_predict()
        
        print("✅ Training completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        logger.error("Training failed due to an unexpected error.", exc_info=True)
        sys.exit(1) 