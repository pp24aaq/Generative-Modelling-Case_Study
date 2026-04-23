import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

device = torch.device("cpu")

# Data Load 
file_path = "Wednesday-workingHours.pcap_ISCX.csv"  # UPDATE THIS

df = pd.read_csv(file_path)

print("Original Shape:", df.shape)

# Clean FData
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Keep only BENIGN + DoS
df = df[df[' Label'].isin(['BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'])]

print("Filtered Shape:", df.shape)

# Features
X = df.drop(columns=[' Label'])

X = X.select_dtypes(include=[np.number])

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

data = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# GAN Models
latent_dim = 32
input_dim = data.shape[1]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

# Training 
criterion = nn.BCELoss()

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50
batch_size = 128

G_losses = []
D_losses = []

# Epochs
for epoch in range(epochs):

    idx = np.random.randint(0, data.shape[0], batch_size)
    real = data[idx]

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Train D
    z = torch.randn(batch_size, latent_dim).to(device)
    fake = G(z)

    loss_real = criterion(D(real), real_labels)
    loss_fake = criterion(D(fake.detach()), fake_labels)
    loss_D = loss_real + loss_fake

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # Train G
    z = torch.randn(batch_size, latent_dim).to(device)
    fake = G(z)

    loss_G = criterion(D(fake), real_labels)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# Synthetic Data
z = torch.randn(1000, latent_dim).to(device)
generated = G(z).detach().cpu().numpy()

real = X_scaled[:1000]

# PCA Visualizaiton
pca = PCA(n_components=2)

real_pca = pca.fit_transform(real)
gen_pca = pca.transform(generated)

plt.figure(figsize=(6,6))
plt.scatter(real_pca[:,0], real_pca[:,1], label="Real", alpha=0.5)
plt.scatter(gen_pca[:,0], gen_pca[:,1], label="Generated", alpha=0.5)
plt.legend()
plt.title("PCA: Real vs Generated Traffic")
plt.show()

# Curves
plt.figure()
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.legend()
plt.title("Training Loss")
plt.show()