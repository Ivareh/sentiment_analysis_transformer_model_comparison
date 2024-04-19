from app.core.model.models import XLNet
from train import TrainXLNet
from app.core.external_dataset_loader import load_data
from app.core import tokenizerXLNet
import preprocess
import evaluation
import os
import torch
import gpu_utilize_testing as gut


torch.cuda.empty_cache()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

XLNET_NUM_TRAIN_EPOCHS = int(os.getenv("XLNET_NUM_TRAIN_EPOCHS"))
XLNET_PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("XLNET_PER_DEVICE_TRAIN_BATCH_SIZE"))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    dataset = load_data.load_imdb_dataset()

    xlnet_tokenized_dataset = preprocess.tokenize_dataset(
        dataset=dataset, tokenizer=tokenizerXLNet
    )
    xlnet_data_collator = preprocess.data_collator(tokenizer=tokenizerXLNet)

    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}

    xlnet_model = XLNet(id2label=id2label, label2id=label2id).get_model()
    
    xlnet_model.to(device)

    num_training_steps = ( len(xlnet_tokenized_dataset["train"]) /  XLNET_PER_DEVICE_TRAIN_BATCH_SIZE ) * XLNET_NUM_TRAIN_EPOCHS
    
    trainXLNet = TrainXLNet(
        tokenizer=tokenizerXLNet,
        tokenized_dataset=xlnet_tokenized_dataset,
        data_collator=xlnet_data_collator,
        compute_metrics=evaluation.compute_metrics,
        xlnet_model=xlnet_model,
        num_training_steps=num_training_steps,
    )

    result = trainXLNet.train_model()
    gut.print_summary(result)


if __name__ == "__main__":
    main()
