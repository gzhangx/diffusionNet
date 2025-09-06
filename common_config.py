import torch
import deepinv

# Device selection
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Model path
MODEL_PATH = "/mnt/e/work/gpvenv/data/diffuOutputs/diffunet_mnist.pth"

# Image size and diffusion steps
IMAGE_SIZE = 32
NUM_DIFFUSION_TIMESTEPS = 1000

# Model creation
def create_model(device):
    model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model
