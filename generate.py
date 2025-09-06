
import torch
import deepinv
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from common_config import get_device, create_model, MODEL_PATH, IMAGE_SIZE, NUM_DIFFUSION_TIMESTEPS

# Settings
n_samples = 8
output_dir = "/mnt/d/temp/generated"
os.makedirs(output_dir, exist_ok=True)

# Device and model
device = get_device()
model = create_model(device)
image_size = IMAGE_SIZE
num_diffusion_timesteps = NUM_DIFFUSION_TIMESTEPS
model_path = MODEL_PATH

# Diffusion params
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_recip_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

# Sampling (reverse diffusion)
def sample(model, n_samples, image_size, num_diffusion_timesteps):
    with torch.no_grad():
        x = torch.randn(n_samples, 1, image_size, image_size, device=device)
        for t in reversed(range(num_diffusion_timesteps)):
            t_tensor = torch.full((n_samples,), t/1000, device=device, dtype=torch.long)
            # Predict noise
            pred_noise = model(x, t_tensor, type_t="timestep")
            # Remove noise
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
        return x

samples = sample(model, n_samples, image_size, num_diffusion_timesteps)

# Save images
for i in range(n_samples):
    save_image(samples[i], f"{output_dir}/sample_{i}.png")
    plt.imshow(samples[i].cpu().squeeze(), cmap="gray")
    plt.title(f"Sample {i}")
    plt.axis('off')
    #plt.show()

print(f"Images saved to {output_dir}")
