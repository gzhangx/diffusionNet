import torch
import deepinv
import os

# Device selection
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

BASE_DIR_PATH = '../trainData/data'
MODEL_PATH = BASE_DIR_PATH + "/diffuOutputs/diffunet_mnist.pth"

# Image size and diffusion steps
IMAGE_SIZE = 32
NUM_DIFFUSION_TIMESTEPS = 1000

# Model creation
def create_model(device):
    model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained model.")
    model.eval()
    return model
