import torch
import torch.nn as nn
from torchvision import models

import numpy as np
from PIL import Image


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        ResNet-based binary classifier.

        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            pretrained: Whether to use pretrained weights
        """
        super(ResNetClassifier, self).__init__()

        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
    def predict_image(self, image, device='cuda'):
        """
        Predict the label for a single image.

        Returns:
            predicted_label: Predicted class (0 or 1)
            confidence: Confidence score for the predicted class
            probabilities: Dictionary with probabilities for both classes
        """
        self.eval()
        print(f"Predicting image... Resized shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

        # Convert to tensor
        image = torch.from_numpy(image).float()  # Now (C, H, W)

        # Apply ImageNet normalization (same as training)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = self.forward(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = output.argmax(1).item()
            confidence = probabilities[0, predicted_label].item()

        # Create probability dictionary
        prob_dict = {
            0: probabilities[0, 0].item(),
            1: probabilities[0, 1].item()
        }

        return predicted_label, confidence, prob_dict