import versioning

version = "v1.0.0"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            # 1st layer
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256 / 2 = 128

            # 2nd layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128 / 2 = 64

            # 3rd layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64 / 2 = 32

            # 4rd layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 / 2 = 16

            nn.Flatten(), #256 * 16 * 16

            # 1st linear
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 2nd linear
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 3nd linear
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 4rd linear
            nn.Linear(64, 16),
            nn.ReLU(),

            # 5th linear
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.model(x)

def load_model():
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    versioning.load_model(model, optimizer, version)

    return model, optimizer

def predict(data, model):
    data = torch.tensor(data, dtype=torch.float32)
    data = data.unsqueeze(0)

    with torch.no_grad():
        model.eval()

        model.to("cpu")

        outputs = model(data)
        
        _, label = torch.max(outputs.data, 1)
    return label.item()

    