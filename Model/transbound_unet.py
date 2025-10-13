#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMSNModel

class TransBound_UNet(nn.Module):
    def __init__(self, img_size=(128, 128), in_channels=1, out_channels=1):
        super(TransBound_UNet, self).__init__()

        self.vit = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
        self.hidden_size = self.vit.config.hidden_size 

        self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1) if in_channels == 1 else nn.Identity()

        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_size, 256, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(16, out_channels, kernel_size=1),  
        )

    def forward(self, x):
        
        x = self.input_conv(x)
        
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)


        vit_outputs = self.vit(pixel_values=x)
        
        x = vit_outputs.last_hidden_state  

        x = x[:, 1:, :]  

        batch_size, seq_len, hidden_size = x.shape
        
        spatial_dim = int(seq_len ** 0.5)
        
        assert spatial_dim ** 2 == seq_len, "ViT output sequence length is not a perfect square."
        
        x = x.permute(0, 2, 1).view(batch_size, hidden_size, spatial_dim, spatial_dim)


        x = self.decoder(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

# test
if __name__ == "__main__":
    model = TransBound_UNet()
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print("Output shape:", y.shape)


