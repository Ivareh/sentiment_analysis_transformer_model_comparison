from transformers import (
    XLNetTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)
from datasets import load_dataset, DatasetDict
from core import tokenizerXLNet
import os


# Tokenizes text and truncate sequences to model's max input length
def _preprocess_function(
    examples: dict, tokenizer: PreTrainedTokenizer
) -> PreTrainedTokenizer:
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenizes dataset
def tokenized_dataset(dataset: DatasetDict) -> DatasetDict:
    tokenized_dataset = dataset.map(_preprocess_function, batched=True)
    return tokenized_dataset


# Create batch of examples using data collator
def data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)
