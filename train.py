import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.gans import GAN16
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def load_batched_npy(file_path):
    """
    Load and concatenate multiple NumPy arrays saved in batches in a single .npy file.
    """
    arrays = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                arrays.append(np.load(f))
            except EOFError:
                break
    return np.concatenate(arrays)

class DistanceMatrixDataset(Dataset):
    def __init__(self, file_path):
        # Load all matrices at once using our helper function
        self.data = load_batched_npy(file_path)
        # Scale down the distances (original values are in Angstroms)
        self.data = self.data / 10.0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        matrix = self.data[idx]
        return torch.FloatTensor(matrix).unsqueeze(0)  # Add channel dimension
    
def save_sample_matrices(gan, epoch, nz, num_samples=10, device='mps'):
    """Generate and save sample matrices periodically"""
    with torch.no_grad():
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        fake_matrices = gan.generator(noise).cpu() * 10.0 # scale back up by factor of 10
        
        # Save the matrices
        save_dir = f'training/samples/epoch_{epoch}'
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(num_samples):
            matrix = fake_matrices[i].squeeze().numpy()
            np.save(f'{save_dir}/sample_{i}.npy', matrix)

def train_gan16(
    train_path='datasets/train/distance_matrices_16.npy',
    batch_size=64,
    nz=100,
    num_epochs=100,
    lr=0.0001,
    beta1=0.5,
    device='mps' if torch.backends.mps.is_available() else 'cpu'
):
    dataset = DistanceMatrixDataset(train_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    gan = GAN16(nz=nz).to(device)
    optimizer_G = optim.Adam(gan.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=lr/10, betas=(beta1, 0.999))
    
    # loss function
    criterion = nn.BCELoss()

    # # Create batch of latent vectors that we will use to visualize generator progression
    # fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # LR schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=5, verbose=True
    )
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_loss = float('inf')
    patience_counter = 0
    # img_list = []
    g_losses = []
    d_losses = []
    
    for epoch in range(num_epochs):
        epoch_g_losses = []
        epoch_d_losses = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, real_matrices in enumerate(pbar):
            real_matrices = real_matrices.to(device)
            batch_size = real_matrices.size(0)

            # generate fake matrices
            noise = torch.randn(batch_size, nz, 1, 1).to(device)
            fake_matrices = gan.generator(noise)

            real_label = torch.ones(batch_size).to(device)
            fake_label = torch.zeros(batch_size).to(device)

            # add noise
            noise_factor = 0.1 * (1 - epoch / num_epochs)
            real_matrices = real_matrices + noise_factor * torch.randn_like(real_matrices)
            fake_matrices = fake_matrices + torch.randn_like(fake_matrices) * noise_factor

            if batch_idx % 3 == 0: # train discriminator less frequently
                # train discriminator
                optimizer_D.zero_grad()
                output_real = gan.discriminator(real_matrices)
                loss_D_real = criterion(output_real, real_label)
                output_fake = gan.discriminator(fake_matrices.detach())
                loss_D_fake = criterion(output_fake, fake_label)
                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                optimizer_D.step()
            
            # train generator
            optimizer_G.zero_grad()
            output_fake = gan.discriminator(fake_matrices)
            loss_G = criterion(output_fake, real_label)
            loss_G.backward()
            optimizer_G.step()

            epoch_g_losses.append(loss_G.item())
            epoch_d_losses.append(loss_D.item())
            
            # update progress
            pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })
        
        # update losses
        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_loss = np.mean(epoch_d_losses)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # update LR schedulers
        scheduler_G.step(avg_g_loss)
        scheduler_D.step(avg_d_loss)

        # early stopping
        current_loss = avg_g_loss + avg_d_loss
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'gan_state_dict': gan.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss_D': avg_d_loss,
                'loss_G': avg_g_loss
            }, 'checkpoints/gan16_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break

    # plot and save losses
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN16 Training Losses')
    plt.legend()
    plt.savefig('training/gan16_losses.png')

if __name__ == '__main__':
    # create directories if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('training', exist_ok=True)
    
    train_gan16()