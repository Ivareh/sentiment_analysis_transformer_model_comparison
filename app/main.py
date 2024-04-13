from train import TrainXLNet
from app.core.external_dataset_loader import load_data
from app.core import tokenizerXLNet
import preprocess
import evaluation


def main():
    dataset = load_data.load_imdb_dataset()

    xlnet_tokenized_dataset = preprocess.tokenize_dataset(
        dataset=dataset, tokenizer=tokenizerXLNet
    )
    xlnet_data_collator = preprocess.data_collator(tokenizer=tokenizerXLNet)

    trainXLNet = TrainXLNet(
        tokenizer=tokenizerXLNet,
        tokenized_dataset=xlnet_tokenized_dataset,
        data_collator=xlnet_data_collator,
        compute_metrics=evaluation.compute_metrics,
    )

    trainXLNet.train_model()


if __name__ == "__main__":
    main()
