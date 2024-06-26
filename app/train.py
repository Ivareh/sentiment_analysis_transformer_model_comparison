import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    XLNetForSequenceClassification,
    get_scheduler,
)
from datasets import DatasetDict
from app.core.model.models import XLNet
from typing import Dict


OUTPUT_DIR = os.getenv("XLNET_OUTPUT_DIR")
LEARNING_RATE = float(os.getenv("XLNET_LEARNING_RATE"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("XLNET_PER_DEVICE_TRAIN_BATCH_SIZE"))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("XLNET_PER_DEVICE_EVAL_BATCH_SIZE"))
NUM_TRAIN_EPOCHS = int(os.getenv("XLNET_NUM_TRAIN_EPOCHS"))
XLNET_WEIGHT_DECAY = float(os.getenv("XLNET_WEIGHT_DECAY"))


class TrainXLNet:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tokenized_dataset: DatasetDict,
        data_collator: DataCollatorWithPadding,
        compute_metrics: Dict,
        xlnet_model: XLNetForSequenceClassification,
        num_training_steps: int,
    ):
        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.xlnet_model = xlnet_model

        self.optimizer = torch.optim.AdamW(
            self.xlnet_model.parameters(), lr=LEARNING_RATE
        )

        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.trainArguments = TrainingArguments(
            output_dir=OUTPUT_DIR,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            weight_decay=XLNET_WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,
        )

    def _create_trainer(self) -> Trainer:
        trainer = Trainer(
            model=self.xlnet_model,
            args=self.trainArguments,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            optimizers=(self.optimizer, self.lr_scheduler),
        )

        return trainer

    def train_model(self):
        trainer = self._create_trainer()

        result = trainer.train()
        return result
