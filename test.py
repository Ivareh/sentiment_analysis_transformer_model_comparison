import torch
import torchvision
import torchaudio

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)