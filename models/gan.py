"""
------------------64 GAN------------------
down-scale factor = 100

--Generator--
nz = 100
ConvTranspose2d( 512, 4, 1, 0)
BatchNorm2d(512)
LeakyReLU(0.2),
ConvTranspose2d(256, 4, 2, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
ConvTranspose2d(128, 4, 2, 1) 
BatchNorm2d(128)
LeakyReLU(0.2)
ConvTranspose2d(64, 4, 2, 1) 
BatchNorm2d(64)
LeakyReLU(0.2)
ConvTranspose2d(1, 4, 2, 1) 
Clamp(>0)
Enforce Symmetry


--Discriminator--

Conv2d(64, 4, 2, 1) 
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(128, 4, 2, 1) 
BatchNorm2d(128)
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(256, 4, 2, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(512, 4, 2, 1)
BatchNorm2d(512)
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(1, 4, 1, 0)
Sigmoid
"""

import torch
import torch.nn as nn

class Generator64(nn.Module):
    def __init__(self, nz=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

    def forward(self, x):
        x = self.main(x)
        x = torch.clamp(x, min=0)  # Enforce non-negativity
        x = 0.5 * (x + x.flip(3))  # Enforce symmetry along the last dimension
        return x

class Discriminator64(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

class GAN64(nn.Module):
    def __init__(self, nz=100):
        super().__init__()
        self.generator = Generator64(nz)
        self.discriminator = Discriminator64()
