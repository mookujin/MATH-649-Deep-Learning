# Mansur Alimbayev  / 31.10.2025
# MATH 649 HW4 (GANs with MNIST database)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Create data loader
batch_size = 128
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

# Display some real images
def show_real_images():
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('real_mnist_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

show_real_images()

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Output: 256 x 4 x 4

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Output: 128 x 8 x 8

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Output: 64 x 16 x 16

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 1 x 32 x 32
        )

    def forward(self, z):
        # Reshape z to (batch_size, latent_dim, 1, 1)
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: 1 x 32 x 32
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 64 x 16 x 16

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 128 x 8 x 8

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 256 x 4 x 4

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, img):
        return self.model(img).view(-1, 1).squeeze(1)

  # Hyperparameters
latent_dim = 100
lr = 0.0002
beta1 = 0.5
num_epochs = 50

# Initialize models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss function and optimizers
criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise for visualization
fixed_noise = torch.randn(64, latent_dim, device=device)

# Lists to track progress
G_losses = []
D_losses = []
D_real_losses = []
D_fake_losses = []

# Training function
def train_gan(num_epochs):
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        start_time = time.time()

        for i, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)

            # Move real images to device and resize to 32x32
            real_imgs = nn.functional.interpolate(real_imgs, size=32).to(device)

            # Create labels
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # ========================
            #  Train Discriminator
            # ========================

            optimizer_D.zero_grad()

            # Train with real images
            outputs_real = discriminator(real_imgs)
            d_loss_real = criterion(outputs_real, real_labels)

            # Train with fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            outputs_fake = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ========================
            #  Train Generator
            # ========================

            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

            # Save losses for plotting
            if i % 50 == 0:
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
                D_real_losses.append(d_loss_real.item())
                D_fake_losses.append(d_loss_fake.item())

        # Print training stats
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Time: {epoch_time:.2f}s')

        # Generate and save sample images
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()

            # Save sample images
            save_generated_images(fake, epoch + 1)

    print("Training completed!")

# Function to save generated images
def save_generated_images(images, epoch, nrow=8):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.imshow(np.transpose(make_grid(images, nrow=nrow, padding=2, normalize=True), (1, 2, 0)))
    ax.set_title(f'Generated Images - Epoch {epoch}')
    plt.savefig(f'generated_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Start training
train_gan(num_epochs)

# Plot training losses
def plot_training_losses():
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(D_real_losses, label='D Real Loss', alpha=0.7)
    plt.plot(D_fake_losses, label='D Fake Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Discriminator Real vs Fake Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_losses()

# Generate final samples
def generate_final_samples():
    # Generate samples from trained generator
    with torch.no_grad():
        z = torch.randn(100, latent_dim, device=device)
        generated_samples = generator(z).detach().cpu()

    # Display generated samples
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_samples[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.suptitle('Final Generated Samples', fontsize=20)
    plt.tight_layout()
    plt.savefig('final_generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

    return generated_samples

final_samples = generate_final_samples()

# Compare real vs generated images
def compare_real_vs_generated():
    # Get real images
    real_iter = iter(train_loader)
    real_images, _ = next(real_iter)
    real_images = nn.functional.interpolate(real_images, size=32)

    # Generate fake images
    with torch.no_grad():
        z = torch.randn(real_images.size(0), latent_dim, device=device)
        fake_images = generator(z).detach().cpu()

    # Create comparison plot
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))

    # Real images
    for i in range(10):
        axes[0, i].imshow(real_images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Real {i}')
        axes[0, i].axis('off')

    # Generated images
    for i in range(10):
        axes[1, i].imshow(fake_images[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Generated {i}')
        axes[1, i].axis('off')

    plt.suptitle('Real vs Generated Images Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('real_vs_generated_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

compare_real_vs_generated()

# Latent space interpolation
def latent_space_interpolation():
    # Generate two random points in latent space
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)

    # Interpolate between them
    steps = 10
    interpolated = []
    for alpha in np.linspace(0, 1, steps):
        z = alpha * z1 + (1 - alpha) * z2
        with torch.no_grad():
            img = generator(z).detach().cpu()
        interpolated.append(img)

    # Display interpolation
    fig, axes = plt.subplots(1, steps, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interpolated[i].squeeze(), cmap='gray')
        ax.set_title(f'Step {i+1}')
        ax.axis('off')

    plt.suptitle('Latent Space Interpolation', fontsize=16)
    plt.tight_layout()
    plt.savefig('latent_space_interpolation.png', dpi=300, bbox_inches='tight')
    plt.show()

latent_space_interpolation()
