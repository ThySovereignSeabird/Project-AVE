from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import csv

import macos_tags
from macos_tags import Color

from tqdm import tqdm  # optional, for progress bar

tensor_transform = transforms.Compose([
    transforms.ToTensor(),  # Scales channels to [0,1] and reorders to (C, H, W)
])

def label_from_tag(fpath):
    tags = macos_tags.get_all(fpath)
    for tag in tags:
        if tag.color == Color.PURPLE:
            return 0
        if tag.color == Color.ORANGE:
            return 1
    return None

class TextureDataset(Dataset):
    def __init__(self, namespaces, transform=tensor_transform):
        self.samples = []
        self.transform = transform

        for namespace in namespaces:
            folder = f"dataset/{namespace}/items/"
            for fname in sorted(f for f in os.listdir(folder) if f.endswith(".png")):
                fpath = os.path.join(folder, fname)
                label = label_from_tag(fpath)
                if (label != None):
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGBA")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

print("Loading dataset...")
namespaces = {"vanilla", "free_use", "asve", "create"}
dataset = TextureDataset(namespaces=namespaces)
print("Finished loading dataset")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class TextureCNN(nn.Module):
    def __init__(self):
        super(TextureCNN, self).__init__()

        # input: 4 channels (RGBA). output: 16 filters = 16 feature maps. 3x3 kernel, padding 1 keeps size same.
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        # 16 channels --> 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # downsample width & height by factor of 2

        self.fc1 = nn.Linear(32 * 4 * 4, 64) # after pooling twice, image is 4x4; 32 feature maps = 32 * 4 * 4 total features
        self.fc2 = nn.Linear(64, 1)
        # pass through a 64-unit hidden layer, then output single value

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16x16 → 8x8
        x = self.pool(F.relu(self.conv2(x)))  # 8x8 → 4x4
        # relu adds nonlinearity, pool downsamples
        x = x.view(-1, 32 * 4 * 4)  # flatten for the dense (fully connected) layer
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # squash domain to binary output range [0, 1]
    
# Training loop

model = TextureCNN() # initialize the model
criterion = nn.BCELoss() # use binary cross-entropy as loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # update weights using gradients

# each batch runs a foward pass, computes loss, clears gradients, runs backpropagation, updates weights
for epoch in range(20):
    for inputs, labels in dataloader:
        outputs = model(inputs).squeeze()  # shape: [batch]
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}") # prints latest loss

# Show results

model.eval()
results = []

with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
        outputs = model(inputs).squeeze()  # shape: [batch]
        probs = outputs.detach().cpu().numpy()  # convert to NumPy array
        labels = labels.cpu().numpy()
        results.extend(zip(probs, labels))

for prob, label in results[:10]:
    print(f"Predicted: {prob:.3f}, Actual: {label}")

with open("output/predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Predicted", "Actual"])
    writer.writerows(results)

torch.save(model.state_dict(), "output/texture_model.pt")