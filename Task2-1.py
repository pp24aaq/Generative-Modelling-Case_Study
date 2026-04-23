
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cpu")

latent_dim = 10
batch_size = 64
epochs = 3000
lr = 0.0005

# 2. Data

def generate_gaussian_data(n=2000):
    centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    data = []

    for cx, cy in centers:
        x = np.random.normal(cx, 0.3, n // 4)
        y = np.random.normal(cy, 0.3, n // 4)
        data.append(np.vstack((x, y)).T)

    return np.vstack(data)

data = generate_gaussian_data()

# Normalize
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std

data = torch.tensor(data, dtype=torch.float32).to(device)

# Model Define

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# 5. Training
G_losses = []
D_losses = []

for epoch in range(epochs):

    idx = np.random.randint(0, data.shape[0], batch_size)
    real = data[idx]

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Train Discriminator
    z = torch.randn(batch_size, latent_dim).to(device)
    fake = G(z)

    loss_real = criterion(D(real), real_labels)
    loss_fake = criterion(D(fake.detach()), fake_labels)
    loss_D = loss_real + loss_fake

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # Train Generator
    z = torch.randn(batch_size, latent_dim).to(device)
    fake = G(z)

    loss_G = criterion(D(fake), real_labels)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# 6. Results
z = torch.randn(1000, latent_dim).to(device)
generated = G(z).detach().cpu().numpy()

# Denormalize
generated = generated * std + mean
real = data.cpu().numpy() * std + mean

# Plots
plt.figure(figsize=(6,6))
plt.scatter(real[:,0], real[:,1], label="Real Data", alpha=0.5)
plt.scatter(generated[:,0], generated[:,1], label="Generated Data", alpha=0.5)
plt.legend()
plt.title("Mixture of Gaussians GAN")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Curves

plt.figure()
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.legend()
plt.title("Training Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()