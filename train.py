import torch
import deepinv
from torchvision import datasets, transforms
import os
from common_config import get_device, create_model, MODEL_PATH, IMAGE_SIZE, NUM_DIFFUSION_TIMESTEPS

batch_size = 64
learn_rate = 1e-5 # tried e3 and it breaks
start_epoch = 0
epochs = 100

device = get_device()
image_size = IMAGE_SIZE
num_diffusion_timesteps = NUM_DIFFUSION_TIMESTEPS
model_path = MODEL_PATH

print("Using device:", device)

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="/mnt/e/work/gpvenv/traindata", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

print(f"Number of batches in train_loader: {len(train_loader)}")
# Or, to get the total number of samples:
print(f"Total number of samples: {len(train_loader.dataset)}")

model = create_model(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
mse = deepinv.loss.MSE()

#https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_recip_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

baseDataPath = os.path.dirname(model_path)
os.makedirs(baseDataPath, exist_ok=True)
checkpoint_path = os.path.join(baseDataPath, "checkpoints")
os.makedirs(checkpoint_path, exist_ok=True)
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
    



for epoch in range(epochs):
    model.train()
    total_loss = 0
    at = 0
    for data, _ in train_loader:
        at = at + 1
        imgs = data.to(device)
        noise = torch.randn_like(imgs)
        #must use imgs.size(0) not batch_size, because last batch can be smaller
        t = torch.randint(0, num_diffusion_timesteps, (imgs.size(0),), device=device)

        noised_imgs = (
            sqrt_alphas_cumprod[t, None, None, None] * imgs
            + sqrt_recip_alphas_cumprod[t, None, None, None] * noise
        )

        optimizer.zero_grad()
        estimated_noise = model(noised_imgs, t, type_t="timestep")  #noise_level
        loss = mse(estimated_noise, noise)
        print('loss ' + str(at)+'/'+str(len(train_loader)), loss.sum(), end='\r')
        loss.sum().backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.sum().item()

    avg_loss = total_loss / len(train_loader)    
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")

    # Save checkpoint
    if (epoch + 1) % 1 == 0:
        print("SAVE")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path+f"/checkpoint_epoch_{epoch+1}.pth")
        torch.save( model.state_dict(), model_path)

torch.save(
    model.state_dict(),
    model_path
)
