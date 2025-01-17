"""
------------------64 Supervised autoencoder(s)------------------

down-scale factor = 100

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
"""

import torch
import torch.nn as nn

class Autoencoder64(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
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
            nn.Dropout(0.1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.clamp(x, min=0)
        x = 0.5 * (x + x.transpose(2, 3))  # Enforce symmetry
        return x



"""
------------------64 Full atom GAN------------------

down-scale factor = 10


--Generator--
nz = 100
ConvTranspose2d( 512, 4, 1, 0)
BatchNorm2d(512)
LeakyReLU(0.2),
ConvTranspose2d(256, 4, 2, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
ConvTranspose2d(128, 4, 4, 0) 
BatchNorm2d(128)
LeakyReLU(0.2)
ConvTranspose2d(64, 4, 4, 0) 
BatchNorm2d(64)
LeakyReLU(0.2)
ConvTranspose2d(1, 4, 2, 1) 
Clamp(>0)
Enforce Symmetry


--Discriminator--

Conv2d(64, 4, 2, 1) 
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(128, 4, 4, 0) 
BatchNorm2d(128)
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(256, 4, 4, 0) 
BatchNorm2d(256)
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(512, 4, 2, 1)
BatchNorm2d(512)
LeakyReLU(0.2)
Dropout(0.1)
Conv2d(1, 4, 1, 0)
Sigmoid



------------------64 Torsion GAN------------------

--Generator--
nz = 100
ConvTranspose1d( 512, 4, 1, 0)
BatchNorm1d(512)
LeakyReLU(0.2),
ConvTranspose1d(256, 4, 2, 1) 
BatchNorm1d(256)
LeakyReLU(0.2)
ConvTranspose1d(128, 4, 2, 1) 
BatchNorm1d(128)
LeakyReLU(0.2)
ConvTranspose1d(64, 4, 2, 1) 
BatchNorm1d(64)
LeakyReLU(0.2)
ConvTranspose1d(1, 4, 2, 1) 
Clamp(>0)
Enforce Symmetry


--Discriminator Scale 64—

Conv1d(64, 4, 2, 1) 
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(128, 4, 2, 1) 
BatchNorm2d(128)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(256, 4, 2, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(512, 4, 2, 1)
BatchNorm2d(512)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(1, 4, 1, 0)
Sigmoid


--Discriminator Scale 16—

Conv1d(64, 3, 1, 1) 
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(128, 4, 2, 1) 
BatchNorm2d(128)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(256, 4, 2, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(512, 3, 1, 1)
BatchNorm2d(512)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(1, 4, 1, 0)
Sigmoid

--Discriminator Scale 8—

Conv1d(64, 3, 1, 1) 
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(128, 4, 2, 1) 
BatchNorm2d(128)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(256, 3, 1, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(512, 4, 2, 1)
BatchNorm2d(512)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(1, 4, 2, 1)
Sigmoid

--Discriminator Scale 4—

Conv1d(64, 3, 1, 1) 
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(128, 3, 1, 1) 
BatchNorm2d(128)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(256, 4, 2, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(1, 4, 2, 1)
Sigmoid

--Discriminator Scale 2—

Conv1d(64, 3, 1, 1) 
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(128, 3, 1, 1) 
BatchNorm2d(128)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(256, 3, 1, 1) 
BatchNorm2d(256)
LeakyReLU(0.2)
Dropout(0.1)
Conv1d(1, 4, 2, 1)
Sigmoid


--Discriminator Scale 1—


Linear(3, 100),
BatchNorm1d(100),
LeakyReLU(0.2),
Dropout(0.1),
Linear(100, 100),
BatchNorm1d(100),
LeakyReLU(0.2),
Dropout(0.1),
Linear(100, 100),
BatchNorm1d(100),
LeakyReLU(0.2),
Dropout(0.1),
Linear(100, 100),
BatchNorm1d(100),
LeakyReLU(0.2),
Dropout(0.1),
Linear(100, 1),
Sigmoid
"""