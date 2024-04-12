import os
from transformers import (
    TrainingArguments,
    Trainer,
)
from xlnet_model import XLNetModel


OUTPUT_DIR = os.getenv("XLNET_OUTPUT_DIR")
LEARNING_RATE = os.getenv("XLNET_LEARNING_RATE")
PER_DEVICE_TRAIN_BATCH_SIZE = os.getenv("XLNET_PER_DEVICE_TRAIN_BATCH_SIZE")
PER_DEVICE_EVAL_BATCH_SIZE = os.getenv("XLNET_PER_DEVICE_EVAL_BATCH_SIZE")
NUM_TRAIN_EPOCHS = os.getenv("XLNET_NUM_TRAIN_EPOCHS")
XLNET_WEIGHT_DECAY = os.getenv("XLNET_WEIGHT_DECAY")


class TrainXLNet:
    def __init__(self):
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
        
        self.trainer = Trainer(
            model=XLNetModel,
            args=self.trainArguments,
            train_dataset=tokenized_imdb["train"],
            eval_dataset=tokenized_imdb["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

