import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
     def __init__(self, num_classes):
         super().__init__()
         self.network = models.resnet18(pretrained=True)
         number_of_features = self.network.fc.in_features
         self.network.fc = nn.Linear(number_of_features, num_classes)
        
     def forward(self, xb):
         return self.network(xb)
    
     def freeze(self):
         for param in self.network.parameters():
             param.requires_grad= False
         for param in self.network.fc.parameters():
             param.requires_grad= True
        
     def unfreeze(self):
         for param in self.network.parameters():
            param.requires_grad= True