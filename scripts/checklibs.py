#Python script to check if libraries imported properly.
#Notice pytorch isn't included. Make sure to download your respective version of Pytorch with the proper CUDA version. 12.x is generally the rule here.
#We recommend using Conda to configure software properly.

import torch
import torchvision
import numpy as np
from PIL import Image
import einops
import matplotlib.pyplot as plt

print("✅ All core imports OK.")
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

