---
tags:
- generated_from_trainer
datasets:
- seccol_events_texts_1500_new
model-index:
- name: Binder
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Binder

This model is a fine-tuned version of [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased) on the seccol_events_texts_1500_new dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 33
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 64
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.24.0
- Pytorch 1.13.0+cu116
- Datasets 2.16.1
- Tokenizers 0.13.3
