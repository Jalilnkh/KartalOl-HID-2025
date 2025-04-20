import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

# ----------------------
# CONFIG
# ----------------------
DATA_DIR = './dataset/GaitDatasetB-silh'
BATCH_SIZE = 8  # reduce batch size for large model
EPOCHS = 20
EMBEDDING_DIM = 1536  # for Swin Transformer Large
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------
# DATASET
# ----------------------
class GaitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for subject_id in sorted(os.listdir(root_dir)):
            subject_path = os.path.join(root_dir, subject_id)
            if not os.path.isdir(subject_path):
                continue
            for subroot, _, files in os.walk(subject_path):
                for file in files:
                    if file.endswith('.png'):
                        img_path = os.path.join(subroot, file)
                        label = int(subject_id)
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Swin-Large expects 384x384
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------------
# MODEL
# ----------------------
class SwinEncoder(nn.Module):
    def __init__(self, embedding_dim=1536):
        super(SwinEncoder, self).__init__()
        self.backbone = create_model('swin_large_patch4_window12_384', pretrained=True)
        self.backbone.head = nn.Identity()
        self.fc = nn.Linear(self.backbone.num_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# ----------------------
# TRAIN
# ----------------------
def train():
    dataset = GaitDataset(DATA_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SwinEncoder(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # for mixed precision

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                feats = model(imgs)
                outputs = nn.functional.normalize(feats, dim=1)
                logits = torch.matmul(outputs, outputs.T)
                mask = torch.eye(logits.size(0), device=DEVICE).bool()
                logits.masked_fill_(mask, -1e9)
                preds = logits.argmax(dim=1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(dataloader):.4f} Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), 'swin_large_encoder.pth')

# ----------------------
# MAIN
# ----------------------
if __name__ == '__main__':
    train()