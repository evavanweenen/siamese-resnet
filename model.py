import torch.nn as nn

from torchvision.models import resnet18, resnet50

class SiameseResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 100)

    def forward(self, anchor, pos, neg):
        return self.cnn(anchor), self.cnn(pos), self.cnn(neg)