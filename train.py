import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from model import GujaratiCNN

# ── Transforms ────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, shear=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ── Load Data ─────────────────────────────────────────────
# IMPORTANT: update these paths to where you extracted the zip
# NEW - CORRECT
train_dataset = datasets.ImageFolder('Datatest\\Train', transform=train_transforms)
val_dataset   = datasets.ImageFolder('Datatest\\Test',  transform=val_transforms)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)

print(f"Classes: {len(train_dataset.classes)}")
print(f"Train images: {len(train_dataset)}")
print(f"Val images:   {len(val_dataset)}")

# ── Model Setup ───────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_classes = len(train_dataset.classes)
model = GujaratiCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── Training Loop ─────────────────────────────────────────
EPOCHS = 30
train_accs, val_accs = [], []

for epoch in range(EPOCHS):

    # Training phase
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_accs.append(train_acc)

    # Validation phase
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_accs.append(val_acc)
    scheduler.step()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {running_loss/len(train_loader):.3f} | "
          f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

# ── Save Everything ───────────────────────────────────────
torch.save(model.state_dict(), 'gujarati_model.pth')
json.dump(train_dataset.classes, open('classes.json', 'w', encoding='utf-8'), ensure_ascii=False)
print("Model saved!")

# ── Plot Accuracy Curve (show in your presentation!) ─────
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs,   label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.title('Gujarati Handwriting CNN - Training Curve')
plt.legend()
plt.savefig('training_curve.png')
plt.show()
print("Chart saved as training_curve.png")