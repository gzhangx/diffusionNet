import torch
import deepinv
import os
from torchvision import datasets, transforms
batch_size = 64

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

TEMP_OUT_DIR = '/mnt/d/temp/denoise_steps'

if get_device() == 'mps':
    TEMP_OUT_DIR = BASE_DIR_PATH + '/temp'

# Image size and diffusion steps
IMAGE_SIZE = 32
NUM_DIFFUSION_TIMESTEPS = 1000

# Model creation
def create_model(device):    

    print("Model loaded successfully!")
    model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(device)
    baseDataPath = os.path.dirname(MODEL_PATH)
    os.makedirs(baseDataPath, exist_ok=True)
    checkpoint_path = os.path.join(baseDataPath, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained model.")
    model.eval()
    return model


transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=BASE_DIR_PATH + "/inputs", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)