"""V15 amendment A1: unpruned accuracy of a checkpoint under Kwon et al.'s own evaluation (evaluate.nlp.test_accuracy
with all-ones masks), so the Phase 1 accuracy *drop* is measured with the same code as the pruned numbers.
Copied into the Kwon repo by v15_kaggle.sh setup and run with its env:
    python v15_kwon_unpruned.py <task> <ckpt_dir>
"""
import sys

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from evaluate.nlp import test_accuracy

task, ckpt = sys.argv[1], sys.argv[2]
config = AutoConfig.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt, config=config).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()
print(f"{task} Unpruned test accuracy: {test_accuracy(model, head_mask, neuron_mask, tokenizer, task):.4f}")
