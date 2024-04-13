import os
from transformers import XLNetForSequenceClassification


MODEL = os.getenv("XLNET_MODEL")


class XLNet:
    def __init__(self, id2label, label2id):
        self.model = XLNetForSequenceClassification.from_pretrained(
            MODEL,
            num_labels=len(self.id2label),
            id2label=id2label,
            label2id=label2id,
        )
