
import torch
import deepinv
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Settings
image_size = 32
num_diffusion_timesteps = 1000
output_dir = "/mnt/d/temp/denoise_steps"
os.makedirs(output_dir, exist_ok=True)
model_path = "/mnt/e/work/gpvenv/data/diffuOutputs/diffunet_mnist.pth"

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Model
model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Diffusion params
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_recip_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

# Load one image from MNIST
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,)),
])
mnist = datasets.MNIST(root="/mnt/e/work/gpvenv/traindata", train=True, download=True, transform=transform)
img, label = mnist[0]  # Pick the first image
img = img.unsqueeze(0).to(device)  # Shape: (1, 1, 32, 32)

# Denoising the selected image and saving intermediate steps
def denoise_steps(model, img, num_diffusion_timesteps, save_every=100):
    with torch.no_grad():
        # Add initial noise
        x = img.clone()
        t_start = num_diffusion_timesteps - 1
        t_tensor = torch.full((1,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x = (
            sqrt_alphas_cumprod[t_start] * x
            + sqrt_recip_alphas_cumprod[t_start] * noise
        )
        steps = []
        for t in reversed(range(num_diffusion_timesteps)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, type_t="timestep")
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod = sqrt_recip_alphas_cumprod[t]
            sqrt_alpha_cumprod = sqrt_alphas_cumprod[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (1 / sqrt_alpha_cumprod) * (x - sqrt_one_minus_alpha_cumprod * pred_noise)
            x += torch.sqrt(betas[t]) * noise
            if t % save_every == 0 or t == num_diffusion_timesteps - 1 or t == 0:
                steps.append((num_diffusion_timesteps - t, x.cpu().squeeze().numpy()))
                plt.imshow(x.cpu().squeeze(), cmap="gray")
                plt.title(f"Step {num_diffusion_timesteps - t}")
                plt.axis('off')
                plt.savefig(f"{output_dir}/step_{num_diffusion_timesteps - t}.png")
                plt.close()
        return steps

steps = denoise_steps(model, img, num_diffusion_timesteps, save_every=100)
print(f"Saved denoising steps to {output_dir}")
