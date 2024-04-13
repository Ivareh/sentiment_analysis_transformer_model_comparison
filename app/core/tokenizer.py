from transformers import XLNetTokenizer
import os

XLNET_MODEL = os.getenv("XLNET_MODEL")

tokenizerXLNet = XLNetTokenizer.from_pretrained(
    XLNET_MODEL
)  # https://huggingface.co/xlnet/xlnet-base-cased. Can also try xlnet-large-cased
