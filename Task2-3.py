import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cpu")

# Data Load
data = np.load("pizza.npy")  # UPDATE PATH

print("Original shape:", data.shape)

# Normalize [0,255] → [-1,1]
data = data / 255.0
data = (data - 0.5) * 2

# Reshape to images
data = data.reshape(-1, 1, 28, 28)

data = torch.tensor(data, dtype=torch.float32)

# GAN Settings 
latent_dim = 100
batch_size = 64
epochs = 30

# GEnerate 
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

# Setup
criterion = nn.BCELoss()

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
G_losses = []
D_losses = []

for epoch in range(epochs):

    idx = np.random.randint(0, data.shape[0], batch_size)
    real = data[idx].to(device)

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Train D
    z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
    fake = G(z)

    loss_real = criterion(D(real), real_labels)
    loss_fake = criterion(D(fake.detach()), fake_labels)
    loss_D = loss_real + loss_fake

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # Train G
    z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
    fake = G(z)

    loss_G = criterion(D(fake), real_labels)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# Images
z = torch.randn(16, latent_dim, 1, 1).to(device)
generated = G(z).detach().cpu()

# Plots
fig, axs = plt.subplots(4, 4, figsize=(6,6))

for i in range(16):
    img = generated[i][0]
    img = (img + 1) / 2  # back to [0,1]
    axs[i//4, i%4].imshow(img, cmap='gray')
    axs[i//4, i%4].axis('off')

plt.suptitle("Generated Pizza Sketches")
plt.show()

# Curves
plt.figure()
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.legend()
plt.title("Training Loss")
plt.show()