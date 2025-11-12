import logging
from typing import Any, List, Dict, Union
from dataclasses import dataclass

import torch
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
import transformers


logger = logging.getLogger(__name__)


@dataclass
class Span:
    type_id: int
    start: int
    end: int
    start_mask: List[int]
    end_mask: List[int]
    span_mask: List[int]


@dataclass
class BinderDataCollator:
    type_input_ids: torch.Tensor
    type_attention_mask: torch.Tensor
    type_token_type_ids: torch.Tensor

    def __post_init__(self):
        self.type_input_ids = torch.tensor(self.type_input_ids)
        self.type_attention_mask = torch.tensor(self.type_attention_mask)
        if self.type_token_type_ids is not None:
            self.type_token_type_ids = torch.tensor(self.type_token_type_ids)

    def __call__(self, features: List) -> Dict[str, Any]:

        # print([len(f['input_ids']) for f in features])

        # exit(0)

        # Dynamically pad sequences in case they are of different lengths
        def _pad(seq: List[int], target_len: int, pad_token: int = 0):
            return seq + [pad_token] * (target_len - len(seq))

        max_len = max(len(f["input_ids"]) for f in features)

        batch = {}
        batch['input_ids'] = torch.tensor([
            _pad(f['input_ids'], max_len, 0) for f in features
        ], dtype=torch.long)

        batch['attention_mask'] = torch.tensor([
            _pad(f['attention_mask'], max_len, 0) for f in features
        ], dtype=torch.bool)

        if "token_type_ids" in features[0]:
            batch['token_type_ids'] = torch.tensor([
                _pad(f['token_type_ids'], max_len, 0) for f in features
            ], dtype=torch.long)

        batch['type_input_ids'] = self.type_input_ids
        batch['type_attention_mask'] = self.type_attention_mask
        if self.type_token_type_ids is not None:
            batch['type_token_type_ids'] = self.type_token_type_ids

        if 'ner' in features[0]:
            # For training
            ner = {}
            # Collate negative mask with shape [batch_size, num_types, ...].
            start_negative_mask, end_negative_mask, span_negative_mask = [], [], []
            # [batch_size, num_types, seq_length]
            start_negative_mask = torch.tensor([f["ner"]["start_negative_mask"] for f in features], dtype=torch.bool)
            end_negative_mask = torch.tensor([f["ner"]["end_negative_mask"] for f in features], dtype=torch.bool)
            # [batch_size, num_types, seq_length, seq_length]
            span_negative_mask = torch.tensor([f["ner"]["span_negative_mask"] for f in features], dtype=torch.bool)
            # Include [CLS]
            start_negative_mask[:, :, 0] = 1
            end_negative_mask[:, :, 0] = 1
            span_negative_mask[:, :, 0, 0] = 1

            ner['start_negative_mask'] =  start_negative_mask
            ner['end_negative_mask'] = end_negative_mask
            ner['span_negative_mask'] = span_negative_mask

            # Collate mention span examples.
            feature_spans = []
            for feature_id, f in enumerate(features):
                spans = []
                for ann in f['ner']['annotations']:
                    type_id, start, end = ann["type_id"], ann["start"], ann["end"]

                    start_mask = start_negative_mask[feature_id][type_id].detach().clone()
                    start_mask[start] = 1

                    end_mask = end_negative_mask[feature_id][type_id].detach().clone()
                    end_mask[end] = 1

                    span_mask = span_negative_mask[feature_id][type_id].detach().clone()
                    span_mask[start][end] = 1

                    spans.append(
                        Span(type_id, start, end, start_mask, end_mask, span_mask)
                    )
                feature_spans.append(spans)

            feature_ids = []
            for feature_id, spans in enumerate(feature_spans):
                feature_ids += [feature_id] * len(spans)
            span_type_ids = [s.type_id for spans in feature_spans for s in spans]

            ner["example_indices"] = [feature_ids, span_type_ids]
            # [batch_size]
            ner["example_starts"] = [s.start for spans in feature_spans for s in spans]
            ner["example_ends"] = [s.end for spans in feature_spans for s in spans]
            # [batch_size, seq_length]
            ner["example_start_masks"] = torch.stack([s.start_mask for spans in feature_spans for s in spans])
            ner["example_end_masks"] = torch.stack([s.end_mask for spans in feature_spans for s in spans])
            # [batch_size, seq_length, seq_length]
            ner["example_span_masks"] = torch.stack([s.span_mask for spans in feature_spans for s in spans])

            batch['ner'] = ner

        return batch


class BinderTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        from transformers.trainer_callback import PrinterCallback
        self.remove_callback(PrinterCallback)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        # print(output)

        # exit(0)

        predictions = self.post_process_function(eval_examples, eval_dataset, output.predictions)
        metrics = predictions["metrics"]

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics


    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)


        transformers.logging.set_verbosity_error()

        output = self.evaluation_loop(
            predict_dataloader,
            description="",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        # print(output)

        # exit(0)

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = predictions["metrics"]

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        return PredictionOutput(predictions=predictions["predictions"], label_ids=predictions["labels"], metrics=metrics)

