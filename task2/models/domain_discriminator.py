import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()

        self.net= nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)