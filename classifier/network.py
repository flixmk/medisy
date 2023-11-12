import torch
import torch.nn as nn
from torchvision import models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomResNet18, self).__init__()
        
        # Load the pre-trained ResNet-18 model from torchvision
        self.resnet = models.resnet18(pretrained=True)
        
        # Remove the last fully-connected layer
        # Note: torchvision's ResNet model has a 'fc' layer as the last layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add a new fully-connected layer
        # The number of input features should match that of the removed layer
        in_features = self.resnet.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # Pass the input tensor through each layer of the model
        x = self.features(x)
        
        # Global average pooling
        x = x.view(x.size(0), -1)
        
        # Classification head
        x = self.fc(x)
        
        return x