import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import pickle
from sklearn.metrics import accuracy_score

# ----------------------
# CONFIG
# ----------------------
DATA_DIR = '../dataset/GaitDatasetB-silh'  # Training data directory
GALLERY_DIR = '../dataset/gallery'         # Gallery directory
PROBE_DIR = '../dataset/probe'             # Probe directory
BATCH_SIZE = 64
EPOCHS = 30
EMBEDDING_DIM = 256
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
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ----------------------
# MODEL
# ----------------------
class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SimpleCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        return self.backbone(x)

# ----------------------
# TRAIN
# ----------------------
def train():
    dataset = GaitDataset(DATA_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SimpleCNN(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            feats = model(imgs)
            outputs = nn.functional.normalize(feats, dim=1)
            logits = torch.matmul(outputs, outputs.T)
            mask = torch.eye(logits.size(0), device=DEVICE).bool()
            logits.masked_fill_(mask, -1e9)
            preds = logits.argmax(dim=1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(dataloader):.4f} Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), 'simple_cnn.pth')

# ----------------------
# LOAD SEQUENCE FROM PKL
# ----------------------
def load_pkl_sequence(pkl_path):
    with open(pkl_path, 'rb') as f:
        sequence = pickle.load(f)
    if isinstance(sequence, dict):
        sequence = sequence['frames']
    tensor_seq = []
    for frame in sequence:
        frame_img = transform(frame).unsqueeze(0)
        tensor_seq.append(frame_img)
    tensor_seq = torch.cat(tensor_seq, dim=0)
    return tensor_seq

# ----------------------
# EXTRACT SEQUENCE FEATURE
# ----------------------
def extract_sequence_feature(model, sequence_tensor):
    model.eval()
    with torch.no_grad():
        sequence_tensor = sequence_tensor.to(DEVICE)
        features = model(sequence_tensor)
        features = nn.functional.normalize(features, dim=1)
        video_feature = features.mean(dim=0)
    return video_feature.cpu()

# ----------------------
# MATCH AND EVALUATE
# ----------------------
def match_and_evaluate():
    model = SimpleCNN(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load('simple_cnn.pth'))

    gallery_features = []
    gallery_labels = []

    for subject_id in sorted(os.listdir(GALLERY_DIR)):
        subject_path = os.path.join(GALLERY_DIR, subject_id)
        for root, _, files in os.walk(subject_path):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_path = os.path.join(root, file)
                    sequence = load_pkl_sequence(pkl_path)
                    feature = extract_sequence_feature(model, sequence)
                    gallery_features.append(feature)
                    gallery_labels.append(int(subject_id))

    probe_features = []
    probe_labels = []

    for subject_id in sorted(os.listdir(PROBE_DIR)):
        subject_path = os.path.join(PROBE_DIR, subject_id)
        for root, _, files in os.walk(subject_path):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_path = os.path.join(root, file)
                    sequence = load_pkl_sequence(pkl_path)
                    feature = extract_sequence_feature(model, sequence)
                    probe_features.append(feature)
                    probe_labels.append(int(subject_id))

    preds = []
    for probe_feat in probe_features:
        dists = torch.norm(torch.stack(gallery_features) - probe_feat, dim=1)
        pred_idx = torch.argmin(dists)
        preds.append(gallery_labels[pred_idx])

    acc = accuracy_score(probe_labels, preds)
    print(f"Identification Accuracy: {acc:.4f}")

# ----------------------
# MAIN
# ----------------------
if __name__ == '__main__':
    train()
    match_and_evaluate()