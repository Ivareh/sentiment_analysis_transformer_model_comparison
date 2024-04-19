import torch
import torchvision
import torchaudio

from app.core.external_dataset_loader import load_data

# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))


# print(torch.__version__)
# print(torchvision.__version__)
# print(torchaudio.__version__)


dataset = load_data.load_imdb_dataset()
load_data.print_dataset_test_example(dataset)