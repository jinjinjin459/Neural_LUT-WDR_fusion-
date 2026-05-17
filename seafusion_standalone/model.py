import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ]
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)

class SeAFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Overexposed feature extractor
        self.overexposed_extractor = nn.Sequential(
            ConvLayer(3, 16),
            ConvLayer(16, 32),
            ConvLayer(32, 64)
        )
        
        # Binary feature extractor
        self.binary_extractor = nn.Sequential(
            ConvLayer(1, 16),
            ConvLayer(16, 32),
            ConvLayer(32, 64)
        )
        
        # Reconstruction
        self.reconstructor = nn.Sequential(
            ConvLayer(128, 64),
            ConvLayer(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, I_gen, I_bin):
        feat_exp = self.overexposed_extractor(I_gen)
        feat_bin = self.binary_extractor(I_bin)
        
        # Binding (Concat on dim=1)
        feat_bound = torch.cat([feat_exp, feat_bin], dim=1)
        
        # Reconstruction
        I_fusion = self.reconstructor(feat_bound)
        return I_fusion

if __name__ == '__main__':
    model = SeAFusion()
    I_gen = torch.randn(8, 3, 400, 600)
    I_bin = torch.randn(8, 1, 400, 600)
    out = model(I_gen, I_bin)
    print(f"I_gen shape: {I_gen.shape}")
    print(f"I_bin shape: {I_bin.shape}")
    print(f"I_fusion shape: {out.shape}")
