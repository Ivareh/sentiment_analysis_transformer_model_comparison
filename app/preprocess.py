from transformers import XLNetTokenizer
from datasets import load_dataset
import os

XLNET_MODEL = os.getenv("XLNET_MODEL")

tokenizer = XLNetTokenizer.from_pretrained(
    XLNET_MODEL
)  # https://huggingface.co/xlnet/xlnet-base-cased. Can also try xlnet-large-cased


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def main():

    imdb = load_dataset("imdb")

    print(imdb["test"][0])

    tokenized_imdb = imdb.map(preprocess_function, batched=True)


if __name__ == "__main__":
    main()
