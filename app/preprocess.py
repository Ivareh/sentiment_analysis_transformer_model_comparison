from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)
from datasets import DatasetDict
from typing import Dict


# Tokenizes text and truncate sequences to model's max input length
def _preprocess_function(
    examples: Dict, tokenizer: PreTrainedTokenizer
) -> PreTrainedTokenizer:
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenizes dataset
def tokenized_dataset(dataset: DatasetDict) -> DatasetDict:
    tokenized_dataset = dataset.map(_preprocess_function, batched=True)
    return tokenized_dataset


# Create batch of examples using data collator
def data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)
