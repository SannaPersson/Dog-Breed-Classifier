import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = torch.hub.load('rwightman/gen-efficientnet-pytorch',
                                       'efficientnet_b3', pretrained=True)
        self.backbone.classifier = nn.Linear(1536, 120)

    def forward(self, x):
        x = self.backbone(x)
        return x


