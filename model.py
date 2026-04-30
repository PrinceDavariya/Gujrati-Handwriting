import torch.nn as nn

class GujaratiCNN(nn.Module):
    def __init__(self, num_classes):
        super(GujaratiCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: detect simple edges and curves
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 64x64 → 32x32

            # Block 2: detect shapes and strokes
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 32x32 → 16x16

            # Block 3: detect complex character parts
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 16x16 → 8x8

            # Block 4: high-level features (needed for 385 classes!)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 8x8 → 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)   # 385 output neurons
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x