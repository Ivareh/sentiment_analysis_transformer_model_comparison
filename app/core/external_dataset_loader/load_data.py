from datasets import DatasetDict, load_dataset


def load_imdb_dataset() -> DatasetDict:
    imdb = load_dataset("imdb")
    return imdb


# Works with DatasetDict type from Hugging Face Datasets library
def print_dataset_test_example(dataset) -> None:
    print(dataset["test"][0])
