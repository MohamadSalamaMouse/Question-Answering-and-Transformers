
The model [roberta-finetuned-subjqa-movies_2](https://huggingface.co/mohamed13579/roberta-finetuned-subjqa-movies_2) is hosted on my Hugging Face account.

# BERT: Bidirectional Encoder Representations from Transformers
![BERT](https://github.com/MohamadSalamaMouse/Question-Answering-and-Transformers/blob/main/1_aqdgJOqpd2Zvv-uMLUJQAw.webp)

BERT (Bidirectional Encoder Representations from Transformers) is a groundbreaking natural language processing (NLP) model introduced by Google AI in 2018. It marked a significant advancement in the field of deep learning for NLP tasks, particularly due to its innovative use of transformers.

## Overview

BERT is based on the transformer architecture, which enables it to capture contextual information from both left and right contexts in a sequence of text. Unlike previous models that processed text input sequentially (either left-to-right or right-to-left), BERT can consider the entire input sentence simultaneously.

## Pre-training

One of the key features of BERT is its pre-training process. It was trained on large amounts of unlabeled text data using two main tasks:
- **Masked Language Model (MLM)**: BERT randomly masks some of the input words and trains the model to predict the masked words based on the surrounding context.
- **Next Sentence Prediction (NSP)**: BERT learns to predict whether a sentence follows another sentence in the input text.

By pre-training on vast amounts of text data, BERT learns rich, contextualized representations of words and sentences.

## Fine-tuning

After pre-training, BERT can be fine-tuned on specific tasks with labeled data. Fine-tuning involves updating the parameters of the pre-trained BERT model to adapt it to the target task. One of the most notable fine-tuning tasks for BERT was on the SQUAD (Stanford Question Answering Dataset) dataset.

## SQUAD Fine-tuning

SQUAD is a dataset consisting of questions posed on Wikipedia articles, where the answer to each question is a segment of text (or span) from the corresponding passage. BERT was fine-tuned on over 100,000 question-answer pairs from the SQUAD dataset.

## State-of-the-Art Performance

Through fine-tuning on various tasks, BERT achieved state-of-the-art performance on a wide range of NLP tasks, including question answering, text classification, named entity recognition, and more. Its versatility and effectiveness have made it one of the most influential models in the NLP community.

## Citation

If you use or reference BERT in your work, please cite the original paper:

[Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.](https://arxiv.org/abs/1810.04805)
## Model Details 
---
license: cc-by-4.0
base_model: deepset/roberta-base-squad2
tags:
- generated_from_trainer
model-index:
- name: roberta-finetuned-subjqa-movies_2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-finetuned-subjqa-movies_2

This model is a fine-tuned version of [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) on the None dataset.

## Model Description

More information needed.

## Intended Uses & Limitations

More information needed.

## Training and Evaluation Data

More information needed.

## Training Procedure

### Training Hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training Results

No specific training results provided.

### Framework Versions

- Transformers 4.38.2
- PyTorch 1.2.0+cu92
- Datasets 1.14.0
- Tokenizers 0.10.3



# Question Answering with Hugging Face pre_trained Model 

This project demonstrates how to use a pre-trained model from Hugging Face for question answering. We'll use the `transformers` library to load a model checkpoint and perform question answering on a given context and question.

## Installation

First, make sure you have the `transformers` library installed. You can install it via pip:

```python
from transformers import pipeline

# Load the question answering pipeline with the specified model checkpoint
model_checkpoint = "mohamed13579/roberta-finetuned-subjqa-movies_2"
question_answerer = pipeline("question-answering", model=model_checkpoint)

# Provide context and question
context = "Replace this with your own context"  # e.g., df_train1.iloc[13].review
question = "Replace this with your question"    # e.g., df_train1.iloc[13].question

# Use the question answering pipeline to get the answer
answer = question_answerer(question=question, context=context)

# Print the answer
print("Question:", question)
print("Answer:", answer['answer'])
