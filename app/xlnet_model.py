import os
from transformers import XLNetForSequenceClassification


MODEL = os.getenv("XLNET_MODEL")

class XLNetModel:
    def __init__(self):
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        self.model = XLNetForSequenceClassification.from_pretrained(MODEL)
