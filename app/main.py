from train import TrainXLNet
from app.core.external_dataset_loader import load_data
from app.core import tokenizerXLNet
import preprocess
import evaluation


def main():
    dataset = load_data.load_imdb_dataset()
    tokenized_dataset = preprocess.tokenized_dataset(dataset)
    data_collator = preprocess.data_collator(tokenizer=tokenizerXLNet)

    trainXLNet = TrainXLNet(
        tokenizer=tokenizerXLNet,
        tokenized_dataset=tokenized_dataset,
        data_collator=data_collator,
        compute_metrics=evaluation.compute_metrics,
    )

    trainXLNet.train_model()


if __name__ == "__main__":
    main()
