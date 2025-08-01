import logging
from typing import Any, List, Dict, Union, Optional
from dataclasses import dataclass

import torch
import numpy as np
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
            # Ensure the sequence is a Python list of ints
            if not isinstance(seq, list):
                # Convert tensors or numpy arrays to list
                if hasattr(seq, "tolist"):
                    seq = seq.tolist()
                else:
                    seq = list(seq)

            # Truncate in the rare case the sequence is already longer than target_len
            if len(seq) >= target_len:
                return seq[:target_len]

            # Pad to the required length
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

        batch_size = len(features)
        
        # Expand type_input_ids to match batch size: (num_types, type_seq_length) -> (batch_size, num_types, type_seq_length)
        batch['type_input_ids'] = self.type_input_ids.unsqueeze(0).repeat(batch_size, 1, 1)
        batch['type_attention_mask'] = self.type_attention_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.type_token_type_ids is not None:
            batch['type_token_type_ids'] = self.type_token_type_ids.unsqueeze(0).repeat(batch_size, 1, 1)

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

    def get_test_dataloader(self, test_dataset):
        """
        Override to disable multiprocessing for test dataloader to avoid "too many open files" error.
        """
        from torch.utils.data import DataLoader
        
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False,
            shuffle=False,
        )

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override the evaluation loop to handle the specific output format of the Binder model.
        """
        import numpy as np
        from transformers.trainer_utils import PredictionOutput
        
        # Call the parent evaluation loop
        output = super().evaluation_loop(
            dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # Process the predictions to ensure they have the correct format
        if output.predictions is None:
            return output
            
        print(f"Debug - output.predictions type: {type(output.predictions)}")
        print(f"Debug - output.predictions length: {len(output.predictions)}")
        
        predictions_list = list(output.predictions)
        print(f"Debug - All predictions:")
        for i, pred in enumerate(predictions_list):
            print(f"  predictions[{i}]: {pred.shape}")
        
        # Initialize variables to store the identified tensors
        start_scores = None
        end_scores = None
        span_scores = None
        
        # Look for tensors with the expected shapes
        # Based on the debug output, we expect:
        # - predictions[0]: (104, 192) - flattened start_scores
        # - predictions[1]: (26, 8, 20, 192) - wrong shape
        # - predictions[2]: (26, 8, 20, 192) - wrong shape  
        # - predictions[3]: (104, 20, 192, 192) - correct span_scores
        
        # First, look for the flattened start_scores (2D tensor)
        for pred in predictions_list:
            if hasattr(pred, 'shape') and len(pred.shape) == 2:
                start_scores = pred
                break
        
        # Look for the correct span_scores (4D tensor with last two dimensions equal)
        for pred in predictions_list:
            if hasattr(pred, 'shape') and len(pred.shape) == 4 and pred.shape[-1] == pred.shape[-2]:
                span_scores = pred
                break
        
        # For end_scores, we need to find a tensor that can be reshaped to match start_scores
        # Look for a tensor with the same first dimension as start_scores
        if start_scores is not None:
            target_features = start_scores.shape[0]
            for pred in predictions_list:
                if hasattr(pred, 'shape') and len(pred.shape) >= 2:
                    # Check if this tensor can be reshaped to match start_scores
                    if pred.shape[0] == target_features or (len(pred.shape) > 2 and pred.shape[0] * pred.shape[1] * pred.shape[2] == target_features):
                        end_scores = pred
                        break
        
        print(f"Debug - Raw predictions shapes:")
        print(f"  start_scores: {start_scores.shape}")
        print(f"  end_scores: {end_scores.shape}")
        print(f"  span_scores: {span_scores.shape}")
        
        # Check if we need to handle the extra batch dimension
        # Note: These are now numpy arrays, not PyTorch tensors
        if hasattr(start_scores, 'ndim') and start_scores.ndim == 4 and start_scores.shape[0] == 1:
            print(f"Debug - Removing extra batch dimension from start_scores and end_scores")
            start_scores = start_scores.squeeze(0)
            end_scores = end_scores.squeeze(0)
            print(f"  After squeeze - start_scores: {start_scores.shape}")
            print(f"  After squeeze - end_scores: {end_scores.shape}")
        
        # If end_scores has the wrong shape, try to reshape it to match start_scores
        if start_scores is not None and end_scores is not None:
            if len(start_scores.shape) == 2 and len(end_scores.shape) > 2:
                target_features = start_scores.shape[0]
                print(f"Debug - Reshaping end_scores from {end_scores.shape} to match {start_scores.shape}")
                # Try to reshape end_scores to match start_scores
                if end_scores.shape[0] * end_scores.shape[1] * end_scores.shape[2] == target_features:
                    end_scores = end_scores.reshape(target_features, end_scores.shape[-1])
                    print(f"  After reshape - end_scores: {end_scores.shape}")
                else:
                    print(f"Debug - Cannot reshape end_scores to match start_scores")
        
        # Handle span_scores shape - the post-processing function expects (features, num_types, seq_len, seq_len)
        if span_scores is not None and len(span_scores.shape) == 4:
            print(f"Debug - span_scores shape: {span_scores.shape}")
            # The post-processing function expects span_scores to have shape (features, num_types, seq_len, seq_len)
            # where each feature has span scores for all entity types
            # So we should keep the current shape as is
            print(f"  Keeping span_scores shape as is for post-processing")
        
        # Convert to numpy arrays (they might already be numpy arrays)
        if hasattr(start_scores, 'cpu'):
            start_scores = start_scores.cpu().numpy()
        if hasattr(end_scores, 'cpu'):
            end_scores = end_scores.cpu().numpy()
        if hasattr(span_scores, 'cpu'):
            span_scores = span_scores.cpu().numpy()

        print(f"Debug - Final numpy shapes:")
        print(f"  start_scores: {start_scores.shape}")
        print(f"  end_scores: {end_scores.shape}")
        print(f"  span_scores: {span_scores.shape}")

        # Debug: Check if we have the right number of predictions for all features
        print(f"Debug - Number of features in dataset: {len(dataloader.dataset)}")
        print(f"Debug - Number of predictions we have: {start_scores.shape[0] if start_scores is not None else 0}")
        
        # If we have fewer predictions than features, we need to handle this
        if start_scores is not None and start_scores.shape[0] < len(dataloader.dataset):
            print(f"Debug - WARNING: We have {start_scores.shape[0]} predictions but {len(dataloader.dataset)} features!")
            print(f"Debug - This suggests we're only processing a subset of features in the evaluation loop")
            print(f"Debug - We need to ensure all features are processed and predictions are collected correctly")
            
            # The issue is that the parent evaluation loop is not processing all features
            # We need to manually process all features to get predictions for all of them
            print(f"Debug - Attempting to process all features manually...")
            
            # Get all predictions from the parent evaluation loop
            all_start_scores = []
            all_end_scores = []
            all_span_scores = []
            
            # Process each batch manually to collect all predictions
            for batch_idx, batch in enumerate(dataloader):
                print(f"Debug - Processing batch {batch_idx + 1}")
                
                # Move batch to the same device as the model
                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = self.model(**batch)
                
                # Extract predictions from outputs
                if isinstance(outputs, tuple):
                    batch_predictions = outputs
                else:
                    batch_predictions = (outputs.start_scores, outputs.end_scores, outputs.span_scores)
                
                # Convert to numpy and collect
                batch_start = batch_predictions[0].cpu().numpy()
                batch_end = batch_predictions[1].cpu().numpy()
                batch_span = batch_predictions[2].cpu().numpy()
                
                all_start_scores.append(batch_start)
                all_end_scores.append(batch_end)
                all_span_scores.append(batch_span)
                
                print(f"Debug - Batch {batch_idx + 1} shapes: {batch_start.shape}, {batch_end.shape}, {batch_span.shape}")
            
            # Concatenate all predictions
            if all_start_scores:
                # For start_scores and end_scores, squeeze the first dimension before concatenating
                # This handles the varying batch sizes issue
                squeezed_start_scores = [scores.squeeze(0) for scores in all_start_scores]
                squeezed_end_scores = [scores.squeeze(0) for scores in all_end_scores]
                
                start_scores = np.concatenate(squeezed_start_scores, axis=0)
                end_scores = np.concatenate(squeezed_end_scores, axis=0)
                span_scores = np.concatenate(all_span_scores, axis=0)
                
                print(f"Debug - After concatenation:")
                print(f"  start_scores: {start_scores.shape}")
                print(f"  end_scores: {end_scores.shape}")
                print(f"  span_scores: {span_scores.shape}")
            else:
                print(f"Debug - No predictions collected, using original predictions")
        
        # Create a new PredictionOutput with the updated predictions
        output = PredictionOutput(
            predictions=(start_scores, end_scores, span_scores),
            label_ids=output.label_ids,
            metrics=output.metrics
        )
        
        return output

