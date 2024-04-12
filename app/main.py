from datasets import load_dataset
from transformers import (
    DataCollatorWithPadding,
    XLNetTokenizer,
    XLNetForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np




data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


model = XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased", num_labels=2, id2label=id2label, label2id=label2id
)


trainer.train()
