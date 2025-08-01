"""
Fine-tune Binder for named entity recognition.
"""

import os
import sys
from pathlib import Path  # more convenient/robust path handling
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
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config import BinderConfig
from src.model import Binder
from src.trainer import BinderDataCollator, BinderTrainer
from src import utils as postprocess_utils

import nltk
import torch

from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer

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

    def __init__(self, config_path = "./conf/inference/text2ner/inference-config.json", device = "auto"):
        os.environ["WANDB_DISABLED"] = "true"

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=config_path)
        if device == "cpu":
            # training_args.no_cuda = True
            # training_args.custom_device = 'cpu'
            training_args.use_cpu = True
        else:
            training_args.use_cpu = False

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        set_seed(training_args.seed)

        self.ru_tokenizer = load("tokenizers/punkt/russian.pickle") # Загрузка токенизатора для русского языка
        self.word_tokenizer = NLTKWordTokenizer()

        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )

        config = BinderConfig(
            pretrained_model_name_or_path=model_args.config_name if model_args.config_name else model_args.model_name_or_path,
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
        model = Binder.from_pretrained(model_args.binder_model_name_or_path, config = config)

        # model.save_pretrained("seccol-binder", safe_serialization = True)
        # model.push_to_hub("seccol-binder")
        # tokenizer.push_to_hub("seccol-binder")

        # exit(0)

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
                "requirement"
            )

        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        self.max_seq_length = max_seq_length

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

        entity_type_id_to_str = [et[data_args.entity_type_key_field] for et in entity_type_knowledge]
        entity_type_str_to_id = {t: i for i, t in enumerate(entity_type_id_to_str)}

        def prepare_type_features(examples):
            tokenized_examples = tokenizer(
                examples[data_args.entity_type_desc_field],
                truncation=True,
                max_length=max_seq_length,
                padding="longest" if len(entity_type_knowledge) <= 1000 else "max_length",
            )
            return tokenized_examples

        with training_args.main_process_first(desc="Tokenizing entity type descriptions"):
            tokenized_descriptions = entity_type_knowledge.map(
                prepare_type_features,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on type descriptions",
                remove_columns=entity_type_knowledge.column_names,
            )

        # Data collator
        data_collator = BinderDataCollator(
            type_input_ids=tokenized_descriptions["input_ids"],
            type_attention_mask=tokenized_descriptions["attention_mask"],
            type_token_type_ids=tokenized_descriptions["token_type_ids"] if "token_type_ids" in tokenized_descriptions else None,
        )

        # Post-processing:
        def post_processing_function(examples, features, predictions, stage=f"eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            metrics = getattr(postprocess_utils, data_args.prediction_postprocess_func)(
                examples=examples,
                features=features,
                predictions=predictions,
                id_to_type=entity_type_id_to_str,
                max_span_length=data_args.max_span_length,
                output_dir=training_args.output_dir if training_args.should_save else None,
                prefix=stage,
                neutral_relative_threshold = data_args.neutral_relative_threshold,
                tokenizer=tokenizer,
                train_file=data_args.train_file,
            )

            return metrics

        # Initialize our Trainer
        self.trainer = BinderTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            eval_examples=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
            post_process_function=post_processing_function,
            compute_metrics=None,
        )

    def map(self, dataset, function, remove_columns):

        with_indices = False
        with_rank = False
        input_columns = None
        batched = True
        batch_size = None
        keep_in_memory = False
        num_proc = 1
        cache_file_name = None
        writer_batch_size = 1000
        features = None
        disable_nullable = False
        fn_kwargs = {}
        suffix_template = "_{rank:05d}_of_{num_proc:05d}"
        drop_last_batch = False
        new_fingerprint = None
        desc = None

        

        """
        Apply a function to all the examples in the table (individually or in batches) and update the table.
        If your function returns a column that already exists, then it overwrites it.

        You can specify whether the function should be batched or not with the `batched` parameter:

        - If batched is `False`, then the function takes 1 example in and should return 1 example.
          An example is a dictionary, e.g. `{"text": "Hello there !"}`.
        - If batched is `True` and `batch_size` is 1, then the function takes a batch of 1 example as input and can return a batch with 1 or more examples.
          A batch is a dictionary, e.g. a batch of 1 example is `{"text": ["Hello there !"]}`.
        - If batched is `True` and `batch_size` is `n > 1`, then the function takes a batch of `n` examples as input and can return a batch with `n` examples, or with an arbitrary number of examples.
          Note that the last batch may have less than `n` examples.
          A batch is a dictionary, e.g. a batch of `n` examples is `{"text": ["Hello there !"] * n}`.
        """



        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]

        if remove_columns is not None:
            missing_columns = set(remove_columns) - set(dataset._data.column_names)
            if missing_columns:
                raise ValueError(
                    f"Column to remove {list(missing_columns)} not in the dataset. Current columns in the dataset: {dataset._data.column_names}"
                )

        dataset_kwargs = {
            "shard": dataset,
            "function": function,
            "with_indices": with_indices,
            "with_rank": with_rank,
            "input_columns": input_columns,
            "batched": batched,
            "batch_size": batch_size,
            "drop_last_batch": drop_last_batch,
            "remove_columns": remove_columns,
            "keep_in_memory": keep_in_memory,
            "writer_batch_size": writer_batch_size,
            "features": features,
            "disable_nullable": disable_nullable,
            "fn_kwargs": fn_kwargs,
        }

        dataset_kwargs["cache_file_name"] = cache_file_name
        transformed_dataset = None

        if transformed_dataset is None:
            # with tqdm(unit = " examples", total = len(dataset), desc = desc or "Map") as pbar:
            for rank, done, content in Dataset._map_single(**dataset_kwargs):
                if done:
                    transformed_dataset = content
                # else:
                #     pbar.update(content)

        assert transformed_dataset is not None, "Failed to retrieve the result from map"

        return transformed_dataset

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

    def predict(self, text):

        trainer = self.trainer
        training_args = self.training_args
        data_args = self.data_args

        offset_mapping = []

        # print("p1")

        sentence_spans = self.ru_tokenizer.span_tokenize(text)

        # print("p2")

        for span in sentence_spans:
            start, end = span
            context = text[start : end]
            word_spans = self.word_tokenizer.span_tokenize(context)
            offset_mapping.extend([(s + start, e + start) for s, e in word_spans])
        start_words, end_words = zip(*offset_mapping)

        # print("p3")

        predict_examples = Dataset.from_dict({
            "text" :  [text],
            "id" : [0],
            "word_start_chars" : [start_words],
            "word_end_chars" : [end_words],
            "entity_types" : [[]],
            "entity_start_chars" : [[]],
            "entity_end_chars" : [[]]
        })

        # print("p4")

        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = self.map(
                predict_examples,
                lambda x: self._prepare_validation_features(x, "test"),
                remove_columns=predict_examples.column_names,
                )

        # print("p5")

        results = trainer.predict(predict_dataset, predict_examples, ignore_keys = ["offset_mapping", "example_id", "split", "token_start_mask", "token_end_mask"])

        # print("p6")
        predictions = results.predictions
        processed_predictions = []
        for p in predictions:
            processed_predictions.append((p.start_char, p.end_char, p.entity_type, p.text))
        processed_predictions = sorted(processed_predictions, key = lambda x : (x[0], x[1]))

        return processed_predictions

from tqdm.auto import tqdm

datasets.disable_progress_bar()
datasets.logging.set_verbosity(datasets.logging.CRITICAL)
tags = set()

if __name__ == "__main__":
    # nltk.download('punkt_tab')

    # --- CONFIGURABLE PATHS -------------------------------------------------
    # You can override these via environment variables if needed.
    inference_config_path   = os.getenv("T2N_INFERENCE_CFG",  "./conf/inference/text2ner-nerel/inference-config.json")
    dataset_path           = os.getenv("T2N_DATASET_PATH",   "/home/student1/data/NEREL1.1/dev")
    predicted_dataset_path = os.getenv("T2N_PRED_OUT",       "/home/student1/data/NEREL1.1/dev_predicted")

    # Ensure the output directory exists
    Path(predicted_dataset_path).mkdir(parents=True, exist_ok=True)

    inf = BinderInference(config_path=inference_config_path)

    dataset_path = Path(dataset_path)

    # -------------------- METRIC HELPERS -----------------------------------
    def compute_tp_fn_fp(predictions, labels):
        if not predictions:
            return {"tp": 0, "fn": len(labels), "fp": 0}
        if not labels:
            return {"tp": 0, "fn": 0, "fp": len(predictions)}
        tp = len(predictions & labels)
        fn = len(labels) - tp
        fp = len(predictions) - tp
        return {"tp": tp, "fn": fn, "fp": fp}

    def compute_precision_recall_f1(tp, fn, fp):
        if tp + fp + fn == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    metrics = {"total": {"tp": 0, "fp": 0, "fn": 0}}

    # -------------------- MAIN LOOP ----------------------------------------
    txt_files = sorted(dataset_path.rglob("*.txt"))
    for txt_path in tqdm(txt_files, desc="Predicting & Scoring"):
        text = txt_path.read_text(encoding="utf-8")
        preds = inf.predict(text)

        # Convert preds to set for metric calc
        pred_set = set(preds)

        # Corresponding golds (if any)
        ann_path = txt_path.with_suffix(".ann")
        golds = []
        if ann_path.exists():
            with ann_path.open("r", encoding="utf-8") as annfile:
                for line in annfile:
                    parts = line.split()
                    if len(parts) > 3 and parts[0].startswith("T"):
                        try:
                            entity_type = parts[1]
                            start_char  = int(parts[2])
                            end_char    = int(parts[3])
                            tags.add(entity_type)
                            golds.append((start_char, end_char, entity_type))
                        except ValueError:
                            continue

        gold_set = set((s, e, t) for s, e, t in golds)

        # ---- WRITE PREDICTED TEXT & ANN -----------------------------------
        out_txt = Path(predicted_dataset_path) / txt_path.name
        out_ann = out_txt.with_suffix(".ann")

        out_txt.write_text(text, encoding="utf-8")
        with out_ann.open("w", encoding="utf-8") as af:
            for idx, (s, e, t, span) in enumerate(sorted(preds)):
                af.write(f"T{idx+1}\t{t} {s} {e}\t{span}\n")

        # ---- UPDATE METRICS ----------------------------------------------
        ts = compute_tp_fn_fp(pred_set, gold_set)
        metrics["total"]["tp"] += ts["tp"]
        metrics["total"]["fp"] += ts["fp"]
        metrics["total"]["fn"] += ts["fn"]

        for entity_type in tags:
            if entity_type not in metrics:
                metrics[entity_type] = {"tp": 0, "fp": 0, "fn": 0}

        for t in tags:
            tagged_golds = set(g for g in gold_set if g[2] == t)
            tagged_preds = set(p for p in pred_set if p[2] == t)
            ts_tag = compute_tp_fn_fp(tagged_preds, tagged_golds)
            metrics[t]["tp"] += ts_tag["tp"]
            metrics[t]["fp"] += ts_tag["fp"]
            metrics[t]["fn"] += ts_tag["fn"]

    # -------------------- FINAL METRIC CALC -------------------------------
    metrics["total"].update(
        compute_precision_recall_f1(metrics["total"]["tp"], metrics["total"]["fn"], metrics["total"]["fp"])
    )

    for t in tags:
        metrics[t].update(
            compute_precision_recall_f1(metrics[t]["tp"], metrics[t]["fn"], metrics[t]["fp"])
        )

    print(metrics)
        


