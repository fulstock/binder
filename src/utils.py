import os
import logging
import collections
import json
import copy
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class Annotation:
    id: str
    entity_type: str
    start_char: int
    end_char: int
    text: str


def compute_tp_fn_fp(predictions: Set, labels: Set, **kwargs) -> Dict[str, float]:
    # tp, fn, fp
    if len(predictions) == 0:
        return {"tp": 0, "fn": len(labels), "fp": 0}
    if len(labels) == 0:
        return {"tp": 0, "fn": 0, "fp": len(predictions)}
    tp = len(predictions & labels)
    fn = len(labels) - tp
    fp = len(predictions) - tp
    return {"tp": tp, "fn": fn, "fp": fp}


def compute_precision_recall_f1(tp: int, fn: int, fp: int, **kwargs) -> Dict[str, float]:
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if tp + fp == 0:
        return {"precision": 1.0, "recall": .0, "f1": .0}
    if tp + fn == 0:
        return {"precision": .0, "recall": 1.0, "f1": .0}
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_everything(tp: int, fn: int, fp: int, **kwargs) -> Dict[str, float]:
    if tp + fp + fn == 0:
        return {"tp" : float(tp), "fp" : float(fp), "fn" : float(fn), "precision": .0, "recall": .0, "f1": .0}
    if tp + fp == 0:
        return {"tp" : float(tp), "fp" : float(fp), "fn" : float(fn), "precision": .0, "recall": .0, "f1": .0}
    if tp + fn == 0:
        return {"tp" : float(tp), "fp" : float(fp), "fn" : float(fn), "precision": .0, "recall": .0, "f1": .0}
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"tp" : float(tp), "fp" : float(fp), "fn" : float(fn), "precision": precision, "recall": recall, "f1": f1}


def postprocess_nested_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    id_to_type: List[str],
    max_span_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    neutral_relative_threshold: Optional[float] = None,
    tokenizer = None,
    **kwargs,
) -> Dict:
    logger.setLevel(log_level)

    # examples -- datasets.Dataset (комменты для val случая)

    # print(examples.column_names)
    # ['text', 'entity_types', 'entity_start_chars', 'entity_end_chars', 'id', 'word_start_chars', 'word_end_chars']
    # print(examples)
    # Dataset({
    #     features: ['text', 'entity_types', 'entity_start_chars', 'entity_end_chars', 'id', 'word_start_chars', 'word_end_chars'],
    #     num_rows: 10
    # })
    # print(features[0].keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'split', 'example_id', 'token_start_mask', 'token_end_mask'])
    # print(features[0])
    # {'input_ids': [101, 34422, 14266, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1], 'offset_mapping': [None, [0, 4], [6, 10], None], 'split': 'dev', 'example_id': '24115799_ru', 'token_start_mask': [0, 1, 1, 0], 'token_end_mask': [0, 1, 0, 0]}

    # exit(1)

    if len(predictions) != 4:
        raise ValueError("`predictions` should be a tuple with four elements (input_ids, start_logits, end_logits, span_logits).")
    all_input_ids, all_start_logits, all_end_logits, all_span_logits = predictions

    if len(predictions[1]) != len(features):
        raise ValueError(f"Got {len(predictions[1])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The gold annotations.
    all_annotations = set()
    # The dictionaries we have to fill.
    all_predictions = set()

    if neutral_relative_threshold is not None:
        all_neutral_predictions = set()
        all_nonzero_predictions = set()

    entity_type_vocab = list(set(id_to_type))
    entity_type_count = collections.defaultdict(int)
    metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    start_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    end_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}

    if neutral_relative_threshold is not None:
        ner_logits = collections.defaultdict(float)
        neu_logits = collections.defaultdict(float)
        all_logits = collections.defaultdict(float)

    # Logging.
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!

    for example_index, example in enumerate(examples):
        example_annotations = set()
        example_predictions = set()
        if neutral_relative_threshold is not None:
            neutral_predictions = set()
            nonzero_predictions = set()
        
        # Looping through all NER annotations.
        for entity_type, start_char, end_char in zip(
            example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
            entity_type_count["all"] += 1
            entity_type_count[entity_type] += 1
            example_annotations.add(Annotation(
                id=example["id"],
                entity_type=entity_type,
                start_char=start_char,
                end_char=end_char,
                text=example["text"][start_char:end_char]
            ))

        
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        # Looping through all the features associated to the current example.

        for feature_index in feature_indices:
            # We grab the masks for start and end indices.

            token_start_mask = np.array(features[feature_index]["token_start_mask"]).astype(bool)
            token_end_mask = np.array(features[feature_index]["token_end_mask"]).astype(bool)

            # We grab the predictions of the model for this feature.
            span_logits = all_span_logits[feature_index]

            # Ensure that the span logits sequence length matches the length of the token masks.
            # Determine the common effective sequence length between logits and token masks.
            seq_len_tokens = len(token_start_mask)
            seq_len_logits = span_logits.shape[1]
            if seq_len_tokens != seq_len_logits:
                common_len = min(seq_len_tokens, seq_len_logits)
                # Truncate both the logits and the masks so that they are compatible.
                span_logits = span_logits[:, :common_len, :common_len]
                token_start_mask = token_start_mask[:common_len]
                token_end_mask = token_end_mask[:common_len]

            ### Two thresholds for flat2nested. Upper and lower one. Upper: positive vs others, lower: neutral vs negative. 

            ### Upper threshold is [CLS] logits as below
            # We use the [CLS] logits as thresholds --- 
            span_preds = np.triu(span_logits > span_logits[:, 0:1, 0:1])

            ### Lower threshold 
            if neutral_relative_threshold is not None:
                span_neutrals = np.triu(span_logits > span_logits[:, 0:1, 0:1] * neutral_relative_threshold)#  & span_logits <= span_logits[:, 0:1, 0:1])

                span_all = np.triu(span_logits > 0)

            type_ids, start_indexes, end_indexes = (
                token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_preds
            ).nonzero()

            if neutral_relative_threshold is not None:
                neutral_type_ids, neutral_start_indexes, neutral_end_indexes = (
                    token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_neutrals
                ).nonzero()
                # neutral_data = (example["id"], span_neutrals)

                all_type_ids, all_start_indexes, all_end_indexes = (
                    token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_all
                ).nonzero()

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all start and end indices.


            for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.


                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider spans with a length that is > max_span_length.
                if end_index - start_index + 1 > max_span_length:
                    continue
                # A prediction contains (example_id, entity_type, start_index, end_index)
                start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
                pred = Annotation(
                    id=example["id"],
                    entity_type=id_to_type[type_id],
                    start_char=start_char,
                    end_char=end_char,
                    text=example["text"][start_char:end_char],
                )
                if neutral_relative_threshold is not None:
                    if id_to_type[type_id] not in ner_logits.keys():
                        ner_logits[id_to_type[type_id]] = collections.defaultdict(float)
                        ner_logits[id_to_type[type_id]][start_char] = collections.defaultdict(float)
                        ner_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    elif start_char not in ner_logits[id_to_type[type_id]].keys():
                        ner_logits[id_to_type[type_id]][start_char] = collections.defaultdict(float)
                        ner_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    elif end_char not in ner_logits[id_to_type[type_id]][start_char].keys():
                        ner_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    ner_logits[id_to_type[type_id]][start_char][end_char] = float(span_logits[type_id][start_index][end_index])
                example_predictions.add(pred)

            if neutral_relative_threshold is not None:
                for type_id, start_index, end_index in zip(neutral_type_ids, neutral_start_indexes, neutral_end_indexes):
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.

                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider spans with a length that is > max_span_length.
                    if end_index - start_index + 1 > max_span_length:
                        continue
                    # A prediction contains (example_id, entity_type, start_index, end_index)
                    start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
                    pred = Annotation(
                        id=example["id"],
                        entity_type=id_to_type[type_id],
                        start_char=start_char,
                        end_char=end_char,
                        text=example["text"][start_char:end_char]
                    )
                    if id_to_type[type_id] not in neu_logits.keys():
                        neu_logits[id_to_type[type_id]] = collections.defaultdict(float)
                        neu_logits[id_to_type[type_id]][start_char] = collections.defaultdict(float)
                        neu_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    elif start_char not in neu_logits[id_to_type[type_id]].keys():
                        neu_logits[id_to_type[type_id]][start_char] = collections.defaultdict(float)
                        neu_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    elif end_char not in neu_logits[id_to_type[type_id]][start_char].keys():
                        neu_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    neu_logits[id_to_type[type_id]][start_char][end_char] = float(span_logits[type_id][start_index][end_index])
                    neutral_predictions.add(pred)
                    
                for type_id, start_index, end_index in zip(all_type_ids, all_start_indexes, all_end_indexes):
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.

                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider spans with a length that is > max_span_length.
                    if end_index - start_index + 1 > max_span_length:
                        continue
                    # A prediction contains (example_id, entity_type, start_index, end_index)
                    start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
                    pred = Annotation(
                        id=example["id"],
                        entity_type=id_to_type[type_id],
                        start_char=start_char,
                        end_char=end_char,
                        text=example["text"][start_char:end_char],
                    )
                    if id_to_type[type_id] not in all_logits.keys():
                        all_logits[id_to_type[type_id]] = collections.defaultdict(float)
                        all_logits[id_to_type[type_id]][start_char] = collections.defaultdict(float)
                        all_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    elif start_char not in all_logits[id_to_type[type_id]].keys():
                        all_logits[id_to_type[type_id]][start_char] = collections.defaultdict(float)
                        all_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    elif end_char not in all_logits[id_to_type[type_id]][start_char].keys():
                        all_logits[id_to_type[type_id]][start_char][end_char] = collections.defaultdict(float)
                    all_logits[id_to_type[type_id]][start_char][end_char] = float(span_logits[type_id][start_index][end_index])
                    nonzero_predictions.add(pred)


        for t in metrics_by_type.keys():

            
            for k, v in compute_tp_fn_fp(
                example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
                example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
            ).items():
                metrics_by_type[t][k] += v
            
     

        all_annotations.update(example_annotations)
        all_predictions.update(example_predictions)
        if neutral_relative_threshold is not None:
            all_neutral_predictions.update(neutral_predictions)
            all_nonzero_predictions.update(nonzero_predictions)
            
        example_gold_starts = set((x.entity_type, x.start_char) for x in example_annotations)
        example_pred_starts = set((x.entity_type, x.start_char) for x in example_predictions)

        for t in start_metrics_by_type.keys():
            
            
            for k, v in compute_tp_fn_fp(
                example_pred_starts if t == "all" else set(filter(lambda x: x[0] == t, example_pred_starts)),
                example_gold_starts if t == "all" else set(filter(lambda x: x[0] == t, example_gold_starts)),
            ).items():
                start_metrics_by_type[t][k] += v
            
       

        example_gold_ends = set((x.entity_type, x.end_char) for x in example_annotations)
        example_pred_ends = set((x.entity_type, x.end_char) for x in example_predictions)

       
        for t in end_metrics_by_type.keys():
            
            for k, v in compute_tp_fn_fp(
                example_pred_ends if t == "all" else set(filter(lambda x: x[0] == t, example_pred_ends)),
                example_gold_ends if t == "all" else set(filter(lambda x: x[0] == t, example_gold_ends)),
            ).items():
                end_metrics_by_type[t][k] += v
           



    metrics = collections.OrderedDict()

    sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
    for x, x_metrics_by_type in {"span": metrics_by_type, "start": start_metrics_by_type, "end": end_metrics_by_type}.items():
        metrics[x] = {}
        for t in sorted_entity_types:
            metrics_for_t = compute_everything(**x_metrics_by_type[t])
            # print(metrics_for_t)
            metrics[x][t] = {}
            for k, v in metrics_for_t.items():
                metrics[x][t][k] = v

    for t in sorted_entity_types:
        support = entity_type_count[t]
        logger.info(f"***** {t} ({support}) *****")
        for x in metrics:
            f1, precision, recall = metrics[x][t]["f1"], metrics[x][t]["precision"], metrics[x][t]["recall"]
            logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (for {x})")

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        if neutral_relative_threshold is not None:
            ner_pred_thresholds = list(span_logits[:, 0:1, 0:1])
            neu_pred_thresholds = list(span_logits[:, 0:1, 0:1] * neutral_relative_threshold)
            logits_file = os.path.join(
                output_dir, "thresholds.json"
            )

            logger.info(f"Saving thresholds to {logits_file}.")
            with open(logits_file, "w", encoding="utf-8") as writer:
                outdict = {}
                outdict["ner"] = dict().copy()
                outdict["neu"] = dict().copy()
                for type_id, threshold in enumerate(ner_pred_thresholds):
                    outdict["ner"][id_to_type[type_id]] = float(threshold)
                for type_id, threshold in enumerate(neu_pred_thresholds):
                    outdict["neu"][id_to_type[type_id]] = float(threshold)
                json.dump(outdict, writer, ensure_ascii = False, indent = 2)

        # Convert flat predictions to hierarchical.
        example_id_to_predictions = {}
        for pred in all_predictions:
            example_id = pred.id
            if example_id not in example_id_to_predictions:
                example_id_to_predictions[example_id] = set()
            if neutral_relative_threshold is not None: 
                example_id_to_predictions[example_id].add((pred.start_char, pred.end_char, pred.entity_type, pred.text, ner_logits[pred.entity_type][pred.start_char][pred.end_char]))
            else:
                example_id_to_predictions[example_id].add((pred.start_char, pred.end_char, pred.entity_type, pred.text))

        if neutral_relative_threshold is not None:
            example_id_to_neutral_predictions = {}
            for neut in all_neutral_predictions:
                example_id = neut.id
                if example_id not in example_id_to_neutral_predictions:
                    example_id_to_neutral_predictions[example_id] = set()
                example_id_to_neutral_predictions[example_id].add((neut.start_char, neut.end_char, neut.entity_type, neut.text, neu_logits[neut.entity_type][neut.start_char][neut.end_char]))

            example_id_to_all_predictions = {}
            for nonzero in all_nonzero_predictions:
                example_id = nonzero.id
                if example_id not in example_id_to_all_predictions:
                    example_id_to_all_predictions[example_id] = set()
                example_id_to_all_predictions[example_id].add((nonzero.start_char, nonzero.end_char, nonzero.entity_type, nonzero.text, all_logits[nonzero.entity_type][nonzero.start_char][nonzero.end_char]))

        predictions_to_save = []
        for example in examples:
            example = copy.deepcopy(example)
            example.pop("word_start_chars")
            example.pop("word_end_chars")
            gold_ner = set()
            for entity_type, start_char, end_char in zip(
                example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
                gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
            example.pop("entity_types")
            example.pop("entity_start_chars")
            example.pop("entity_end_chars")

            pred_ner = example_id_to_predictions.get(example["id"], set())
            example["gold_ner"] = sorted(gold_ner)
            example["pred_ner"] = sorted(pred_ner)
            predictions_to_save.append(example)        

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            for pred in predictions_to_save:
                writer.write(json.dumps(pred, ensure_ascii = False) + "\n")

        if neutral_relative_threshold is not None:         
            neutrals_to_save = []
            for example in examples:
                example = copy.deepcopy(example)
                example.pop("word_start_chars")
                example.pop("word_end_chars")
                gold_ner = set()
                for entity_type, start_char, end_char in zip(
                    example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
                    gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
                example.pop("entity_types")
                example.pop("entity_start_chars")
                example.pop("entity_end_chars")

                pred_ner = example_id_to_neutral_predictions.get(example["id"], set())
                example["gold_ner"] = sorted(gold_ner)
                example["pred_ner"] = sorted(pred_ner)
                neutrals_to_save.append(example)

            neutrals_file = os.path.join(
                output_dir, "neutrals.json" if prefix is None else f"{prefix}_neutrals.json"
            )

            logger.info(f"Saving neutrals to {neutrals_file}.")

            with open(neutrals_file, "w", encoding="utf-8") as writer:
                for neut in neutrals_to_save:
                    writer.write(json.dumps(neut, ensure_ascii = False) + "\n")

            logits_to_save = []
            for example in examples:
                example = copy.deepcopy(example)
                example.pop("word_start_chars")
                example.pop("word_end_chars")
                gold_ner = set()
                for entity_type, start_char, end_char in zip(
                    example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
                    gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
                example.pop("entity_types")
                example.pop("entity_start_chars")
                example.pop("entity_end_chars")

                pred_ner = example_id_to_all_predictions.get(example["id"], set())
                example["gold_ner"] = sorted(gold_ner)
                example["pred_ner"] = sorted(pred_ner)
                logits_to_save.append(example)        

            all_logits_file = os.path.join(
                output_dir, "all_logits.json" if prefix is None else f"{prefix}_all_logits.json"
            )

            logger.info(f"Saving all logits to {all_logits_file}.")
            with open(all_logits_file, "w", encoding="utf-8") as writer:
                for logit in logits_to_save:
                    writer.write(json.dumps(logit, ensure_ascii = False) + "\n")

        metric_file = os.path.join(
            output_dir, "metrics.json" if prefix is None else f"{prefix}_metrics.json"
        )

        logger.info(f"Saving metrics to {metric_file}.")
        with open(metric_file, "a", encoding="utf-8") as writer:
            writer.write(json.dumps(metrics) + "\n")

    reduced_metrics = dict()

    for tag in metrics["span"].keys():
        if tag != 'all':
            reduced_metrics[tag + "-f1"] = metrics["span"][tag]["f1"]
            reduced_metrics[tag + "-recall"] = metrics["span"][tag]["recall"]
            reduced_metrics[tag + "-precision"] = metrics["span"][tag]["precision"]
            reduced_metrics[tag + "-tp"] = metrics["span"][tag]["tp"]
            reduced_metrics[tag + "-fp"] = metrics["span"][tag]["fp"]
            reduced_metrics[tag + "-fn"] = metrics["span"][tag]["fn"]

    reduced_metrics["macro-f1"] = reduced_metrics["f1"] = np.mean([v for k, v in reduced_metrics.items() if "f1" in k])
    reduced_metrics["macro-precision"] = reduced_metrics["precision"] = np.mean([v for k, v in reduced_metrics.items() if "precision" in k])
    reduced_metrics["macro-recall"] = reduced_metrics["recall"] = np.mean([v for k, v in reduced_metrics.items() if "recall" in k])



    total_tp = sum([v for k, v in reduced_metrics.items() if "-tp" in k])
    total_fp = sum([v for k, v in reduced_metrics.items() if "-fp" in k])
    total_fn = sum([v for k, v in reduced_metrics.items() if "-fn" in k])

    prf = compute_precision_recall_f1(int(total_tp), int(total_fn), int(total_fp))

    reduced_metrics["micro-precision"] = prf["precision"]
    reduced_metrics["micro-recall"] = prf["recall"]
    reduced_metrics["micro-f1"] = prf["f1"] 

    metrics = reduced_metrics

    return {
        "predictions": all_predictions,
        "labels": all_annotations,
        "metrics": metrics,
    }


def search_insert(array: List, x) -> int:
    l, r = 0, len(array) - 1
    while l <= r:
        m = (l + r) // 2
        if array[m] == x:
            return m
        elif array[m] < x:
            l = m + 1
        else:
            r = m - 1
    return l


def remove_overlaps(pred_scores: Dict[Annotation, float]) -> Set:
    predictions, starts = [], []
    for pred in sorted(pred_scores, key=lambda x: (-pred_scores[x], x.start_char, x.end_char)):
        start, end = pred.start_char, pred.end_char
        if len(predictions) == 0:
            predictions.append(pred)
            starts.append(start)
        else:
            index = search_insert(starts, start)
            if index == 0:
                next_start = predictions[index].start_char
                if end <= next_start:
                    predictions.insert(index, pred)
                    starts.insert(index, start)
            elif index == len(predictions):
                prev_end = predictions[index - 1].end_char
                if start >= prev_end:
                    predictions.insert(index, pred)
                    starts.insert(index, start)
            else:
                next_start = predictions[index].start_char
                prev_end = predictions[index - 1].end_char
                if start >= prev_end and end <= next_start:
                    predictions.insert(index, pred)
                    starts.insert(index, start)
    return set(predictions)


def convert_to_iob(
    text: str,
    word_start_chars: List[int],
    word_end_chars: List[int],
    entity_start_chars: List[int],
    entity_end_chars: List[int],
    entity_types: List[str],
    **kwargs
) -> Dict[str, List[str]]:
    words = [text[s:e] for s, e in zip(word_start_chars, word_end_chars)]
    labels = []
    pos = 0
    while pos < len(word_start_chars):
        start, end = word_start_chars[pos], word_end_chars[pos]
        if start in entity_start_chars:
            index = entity_start_chars.index(start)
            labels.append("B-" + entity_types[index])
            if end in entity_end_chars:
                assert index == entity_end_chars.index(end), breakpoint()
            else:
                while end not in entity_end_chars:
                    pos += 1
                    start, end = word_start_chars[pos], word_end_chars[pos]
                    assert start not in entity_start_chars, breakpoint()
                    labels.append("I-" + entity_types[index])
                assert index == entity_end_chars.index(end), breakpoint()
        else:
            labels.append("O")
        pos += 1
    assert len(words) == len(labels), breakpoint()
    return {"words": words, "labels": labels}


def error_analysis(annotations: Set[Annotation], predictions: Set[Annotation], entity_types: List[str]) -> Dict:
    fp = [p for p in predictions if p not in annotations]
    fn = [a for a in annotations if a not in predictions]
    fp_counter = collections.Counter([(x.text, x.entity_type) for x in fp])
    fn_counter = collections.Counter([(x.text, x.entity_type) for x in fn])
    fp_patterns = '|'.join(sorted(set([k[0].lower() for k, _ in fp_counter.most_common(30)])))
    ret = collections.OrderedDict({
        "most-common fp patterns": fp_patterns,
        "most-common fp errors": {" | ".join(k): v for k, v in fp_counter.most_common(30)},
        "most-common fn errors": {" | ".join(k): v for k, v in fn_counter.most_common(30)},
    })
    for t in entity_types:
        fp_counter = collections.Counter([x.text for x in fp if x.entity_type == t])
        fn_counter = collections.Counter([x.text for x in fn if x.entity_type == t])
        fp_patterns = '|'.join(sorted(set([k.lower() for k, _ in fp_counter.most_common(10)])))
        ret.update({
            t: {
                "most-common fp patterns": fp_patterns,
                "most-common fp errors": {k: v for k, v in fp_counter.most_common(10)},
                "most-common fn errors": {k: v for k, v in fn_counter.most_common(10)},
            }
        })
    return ret


def postprocess_nested_predictions_with_threshold(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    id_to_type: List[str],
    max_span_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    neutral_relative_threshold: Optional[float] = None,
    tokenizer = None,
    threshold_factor: float = 1.0,
    **kwargs,
) -> Dict:
    """
    Custom postprocessing function that supports threshold factor for lowering prediction thresholds.
    """
    logger.setLevel(log_level)
    
    if len(predictions) != 4:
        raise ValueError("`predictions` should be a tuple with four elements (input_ids, start_logits, end_logits, span_logits).")
    all_input_ids, all_start_logits, all_end_logits, all_span_logits = predictions

    if len(predictions[1]) != len(features):
        raise ValueError(f"Got {len(predictions[1])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The gold annotations.
    all_annotations = set()
    # The dictionaries we have to fill.
    all_predictions = set()

    entity_type_vocab = list(set(id_to_type))
    entity_type_count = collections.defaultdict(int)
    metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}

    # Logging.
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features with threshold factor {threshold_factor}.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        example_annotations = set()
        example_predictions = set()
        
        # Looping through all NER annotations.
        for entity_type, start_char, end_char in zip(
            example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
            entity_type_count["all"] += 1
            entity_type_count[entity_type] += 1
            example_annotations.add(Annotation(
                id=example["id"],
                entity_type=entity_type,
                start_char=start_char,
                end_char=end_char,
                text=example["text"][start_char:end_char]
            ))

        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the masks for start and end indices.
            token_start_mask = np.array(features[feature_index]["token_start_mask"]).astype(bool)
            token_end_mask = np.array(features[feature_index]["token_end_mask"]).astype(bool)

            # We grab the predictions of the model for this feature.
            span_logits = all_span_logits[feature_index]

            # Ensure that the span logits sequence length matches the length of the token masks.
            # Determine the common effective sequence length between logits and token masks.
            seq_len_tokens = len(token_start_mask)
            seq_len_logits = span_logits.shape[1]
            if seq_len_tokens != seq_len_logits:
                common_len = min(seq_len_tokens, seq_len_logits)
                # Truncate both the logits and the masks so that they are compatible.
                span_logits = span_logits[:, :common_len, :common_len]
                token_start_mask = token_start_mask[:common_len]
                token_end_mask = token_end_mask[:common_len]

            ### Two thresholds for flat2nested. Upper and lower one. Upper: positive vs others, lower: neutral vs negative. 

            ### Upper threshold is [CLS] logits as below
            # We use the [CLS] logits as thresholds --- 
            span_preds = np.triu(span_logits > span_logits[:, 0:1, 0:1])

            ### Lower threshold 
            if neutral_relative_threshold is not None:
                span_neutrals = np.triu(span_logits > span_logits[:, 0:1, 0:1] * neutral_relative_threshold)#  & span_logits <= span_logits[:, 0:1, 0:1])

                span_all = np.triu(span_logits > 0)

            type_ids, start_indexes, end_indexes = (
                token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_preds
            ).nonzero()

            if neutral_relative_threshold is not None:
                neutral_type_ids, neutral_start_indexes, neutral_end_indexes = (
                    token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_neutrals
                ).nonzero()
                # neutral_data = (example["id"], span_neutrals)

                all_type_ids, all_start_indexes, all_end_indexes = (
                    token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_all
                ).nonzero()

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all start and end indices.
            for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider spans with a length that is > max_span_length.
                if end_index - start_index + 1 > max_span_length:
                    continue
                # A prediction contains (example_id, entity_type, start_index, end_index)
                start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
                pred = Annotation(
                    id=example["id"],
                    entity_type=id_to_type[type_id],
                    start_char=start_char,
                    end_char=end_char,
                    text=example["text"][start_char:end_char],
                )
                example_predictions.add(pred)

        for t in metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
                example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
            ).items():
                metrics_by_type[t][k] += v

        all_annotations.update(example_annotations)
        all_predictions.update(example_predictions)

    metrics = collections.OrderedDict()
    sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
    
    metrics["span"] = {}
    for t in sorted_entity_types:
        metrics_for_t = compute_everything(**metrics_by_type[t])
        metrics["span"][t] = {}
        for k, v in metrics_for_t.items():
            metrics["span"][t][k] = v

    for t in sorted_entity_types:
        support = entity_type_count[t]
        logger.info(f"***** {t} ({support}) *****")
        f1, precision, recall = metrics["span"][t]["f1"], metrics["span"][t]["precision"], metrics["span"][t]["recall"]
        logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (with threshold factor {threshold_factor})")

    reduced_metrics = dict()
    for tag in metrics["span"].keys():
        if tag != 'all':
            reduced_metrics[tag + "-f1"] = metrics["span"][tag]["f1"]
            reduced_metrics[tag + "-recall"] = metrics["span"][tag]["recall"]
            reduced_metrics[tag + "-precision"] = metrics["span"][tag]["precision"]
            reduced_metrics[tag + "-tp"] = metrics["span"][tag]["tp"]
            reduced_metrics[tag + "-fp"] = metrics["span"][tag]["fp"]
            reduced_metrics[tag + "-fn"] = metrics["span"][tag]["fn"]

    reduced_metrics["macro-f1"] = reduced_metrics["f1"] = np.mean([v for k, v in reduced_metrics.items() if "f1" in k])
    reduced_metrics["macro-precision"] = reduced_metrics["precision"] = np.mean([v for k, v in reduced_metrics.items() if "precision" in k])
    reduced_metrics["macro-recall"] = reduced_metrics["recall"] = np.mean([v for k, v in reduced_metrics.items() if "recall" in k])

    total_tp = sum([v for k, v in reduced_metrics.items() if "-tp" in k])
    total_fp = sum([v for k, v in reduced_metrics.items() if "-fp" in k])
    total_fn = sum([v for k, v in reduced_metrics.items() if "-fn" in k])

    prf = compute_precision_recall_f1(int(total_tp), int(total_fn), int(total_fp))

    reduced_metrics["micro-precision"] = prf["precision"]
    reduced_metrics["micro-recall"] = prf["recall"]
    reduced_metrics["micro-f1"] = prf["f1"] 

    metrics = reduced_metrics

    return {
        "predictions": all_predictions,
        "labels": all_annotations,
        "metrics": metrics,
    }
