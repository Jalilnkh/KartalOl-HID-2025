import os
import torch
import torch.nn as nn
import pickle
import pandas as pd
from torchvision import transforms
from model import SimpleCNN  # Assume you have SimpleCNN defined somewhere
from tqdm import tqdm

# ----------------------
# CONFIG
# ----------------------
GALLERY_DIR = '../dataset/gallery'  # path after decompression
PROBE_DIR = '../dataset/probe'      # path after decompression
OUTPUT_CSV = '../results/jalil/submission.csv'     # final output
EMBEDDING_DIM = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()
])

# ----------------------
# LOAD MODEL
# ----------------------
model = SimpleCNN(embedding_dim=EMBEDDING_DIM).to(DEVICE)
model.load_state_dict(torch.load('Baseline.pth', map_location=DEVICE))
model.eval()

# ----------------------
# LOAD PKL SEQUENCE
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
# EXTRACT FEATURE
# ----------------------
def extract_sequence_feature(sequence_tensor):
    with torch.no_grad():
        sequence_tensor = sequence_tensor.to(DEVICE)
        features = model(sequence_tensor)
        features = nn.functional.normalize(features, dim=1)
        video_feature = features.mean(dim=0)
    return video_feature.cpu()

# ----------------------
# PREPARE GALLERY FEATURES
# ----------------------
gallery_features = []
gallery_labels = []

print("Extracting gallery features...")
for subject_id in tqdm(sorted(os.listdir(GALLERY_DIR))):
    subject_path = os.path.join(GALLERY_DIR, subject_id)
    for root, _, files in os.walk(subject_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_path = os.path.join(root, file)
                sequence = load_pkl_sequence(pkl_path)
                feature = extract_sequence_feature(sequence)
                gallery_features.append(feature)
                gallery_labels.append(subject_id)

gallery_features = torch.stack(gallery_features)

# ----------------------
# INFERENCE ON PROBE
# ----------------------
submission = []

print("Processing probe set...")
for subject_id in tqdm(sorted(os.listdir(PROBE_DIR))):
    subject_path = os.path.join(PROBE_DIR, subject_id)
    for root, _, files in os.walk(subject_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_path = os.path.join(root, file)
                sequence = load_pkl_sequence(pkl_path)
                feature = extract_sequence_feature(sequence)
                dists = torch.norm(gallery_features - feature, dim=1)
                pred_idx = torch.argmin(dists)
                pred_label = gallery_labels[pred_idx]
                submission.append([file.replace('.pkl',''), pred_label])

# ----------------------
# SAVE SUBMISSION FILE
# ----------------------
submission_df = pd.DataFrame(submission, columns=['file', 'label'])
submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
