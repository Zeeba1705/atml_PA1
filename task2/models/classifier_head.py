import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        return self.fc(x)