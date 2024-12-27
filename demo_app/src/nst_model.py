import torch
import torch.nn as nn
from torchvision import models


class NST_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_content_idx = [21]
        self.layer_style_idx = [0, 5, 10, 19, 28]
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:29]

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        content_features = []
        style_features = []

        for layer_idx, layer in enumerate(self.model):
            x = layer(x)

            if layer_idx in self.layer_content_idx:
                content_features.append(x)

            if layer_idx in self.layer_style_idx:
                style_features.append(x)

        return content_features, style_features
