import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.gans import GAN16
from tqdm import tqdm

class DistanceMatrixDataset(Dataset):
    def __init__(self, file_path):
        # load and concatenate batched arrays
        with open(file_path, 'rb') as f:
            arrays = []
            while True:
                try:
                    arrays.append(np.load(f))
                except ValueError:
                    break
        self.data = np.concatenate(arrays)
        # scale down by factor of 10 (as mentioned in the GAN16 comments)
        self.data = self.data / 10.0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # add channel dimension and convert to tensor
        return torch.FloatTensor(self.data[idx]).unsqueeze(0)

def train_gan16(
    train_path='datasets/train/distance_matrices_16.npy',
    batch_size=64,
    nz=100,
    num_epochs=100,
    lr=0.0002,
    beta1=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    dataset = DistanceMatrixDataset(train_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    gan = GAN16(nz=nz).to(device)
    optimizer_G = optim.Adam(gan.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # loss function
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for real_matrices in pbar:
            batch_size = real_matrices.size(0)
            real_matrices = real_matrices.to(device)
            
            # labels for real and fake data
            real_label = torch.ones(batch_size).to(device)
            fake_label = torch.zeros(batch_size).to(device)
            
            # train discriminator
            optimizer_D.zero_grad()
            output_real = gan.discriminator(real_matrices)
            loss_D_real = criterion(output_real, real_label)
            
            noise = torch.randn(batch_size, nz, 1, 1).to(device)
            fake_matrices = gan.generator(noise)
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
            
            # update progress
            pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })
        
        torch.save({
            'epoch': epoch,
            'gan_state_dict': gan.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, f'checkpoints/gan16_epoch_{epoch+1}.pt')

if __name__ == '__main__':
    # create checkpoints directory if it doesn't exist
    # import os
    # os.makedirs('checkpoints', exist_ok=True)
    
    train_gan16()