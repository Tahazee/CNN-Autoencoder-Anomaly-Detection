import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Symmetric Convolutional Autoencoder with dense semantic bottleneck
    designed for unsupervised structural anomaly detection.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        
        # Encoder: 256x256 -> 16x16
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # -> 128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # -> 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),          # -> 16x16
            nn.ReLU(inplace=True)
        )
        
        # Dense Bottleneck
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        self.unflatten = nn.Unflatten(1, (256, 16, 16))
        
        # Decoder: 16x16 -> 256x256
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # -> 256x256
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_conv(x)
        flattened = self.flatten(encoded)
        latent = self.fc_encode(flattened)
        
        reconstructed_latent = self.fc_decode(latent)
        unflattened = self.unflatten(reconstructed_latent)
        return self.decoder_conv(unflattened)