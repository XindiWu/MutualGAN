
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes=2):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
