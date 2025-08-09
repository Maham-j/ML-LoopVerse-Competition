## 1.	Data Cleaning and Preprocessing 
#files upload
from google.colab import files
uploaded = files.upload()  

#unzip
!unzip /content/EuroSAT_RGB.zip -d /content/EuroSAT_RGB

#check data 
import os

data_path = '/content/EuroSAT_RGB'  
print("Listing first 10 files in dataset folder:")
print(os.listdir(data_path)[:10])


import os
import cv2
import numpy as np
from glob import glob

data_path = '/content/EuroSAT_RGB' 
cleaned_path = '/content/cleaned_dataset'
os.makedirs(cleaned_path, exist_ok=True)

image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
IMG_SIZE = (224, 224)

def is_corrupted_or_noisy(img):
    if img is None:
        return True  

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size

    # Black patches check: allow up to 60% black pixels
    black_pixels = np.sum(gray == 0)
    if black_pixels / total_pixels > 0.6:
        return True

    # Laplacian variance: blurry if less than 4, noisy if more than 2000
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 4:
        return True
    if laplacian_var > 2000:
        return True

    # Contrast check: std dev less than 5 considered low contrast
    if np.std(gray) < 5:
        return True

    return False

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if is_corrupted_or_noisy(img):
        return None
    img = cv2.resize(img, IMG_SIZE)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.medianBlur(img, 3)
    return img

processed_count = 0
corrupted_count = 0

for ext in image_extensions:
    files = glob(os.path.join(data_path, f'**/*{ext}'), recursive=True)
    for file in files:
        img_clean = preprocess_image(file)
        if img_clean is not None:
            relative_path = os.path.relpath(file, data_path)
            save_path = os.path.join(cleaned_path, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_clean)
            processed_count += 1
        else:
            corrupted_count += 1

print(f"Processed images saved: {processed_count}")
print(f"Corrupted/unreadable images skipped: {corrupted_count}")


##count of images in folder
import os

def count_images(folder):
    exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    count = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                count += 1
    return count

cleaned_folder = '/content/cleaned_dataset'
corrupted_folder = '/content/corrupted_dataset'

cleaned_count = count_images(cleaned_folder)
corrupted_count = count_images(corrupted_folder)

print(f"Cleaned images count: {cleaned_count}")
print(f"Corrupted images count: {corrupted_count}")
print(f"Total images: {cleaned_count + corrupted_count}")





##2.	Feature Extraction 
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from skimage.feature import local_binary_pattern

cleaned_path = '/content/cleaned_dataset/EuroSAT_RGB'

# List classes (subfolders)
classes = [d for d in os.listdir(cleaned_path) if os.path.isdir(os.path.join(cleaned_path, d))]
print("Classes found:", classes)

def extract_color_histogram(image):
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    hist = []
    for chan, color in zip(chans, colors):
        h = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist.append(h)
    return hist

def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def extract_lbp(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize histogram
    return hist

def plot_features(image, histograms, edges, lbp_hist, class_name):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image - {class_name}")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    for i, color in enumerate(['b', 'g', 'r']):
        plt.plot(histograms[i], color=color)
    plt.title('Color Histogram')
    plt.xlim([0, 256])

    plt.subplot(1, 4, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Map (Canny)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.bar(range(len(lbp_hist)), lbp_hist)
    plt.title('LBP Histogram')
    plt.xlabel('LBP Pattern')
    plt.ylabel('Normalized Frequency')

    plt.show()

# Pick one random image per class, extract features and plot
for class_name in classes:
    class_folder = os.path.join(cleaned_path, class_name)
    images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not images:
        continue

    sample_img_name = choice(images)
    img_path = os.path.join(class_folder, sample_img_name)
    img = cv2.imread(img_path)

    color_hist = extract_color_histogram(img)
    edges = extract_edges(img)
    lbp_hist = extract_lbp(img)

    plot_features(img, color_hist, edges, lbp_hist, class_name)


##diff of both cleaned and raw

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern

def extract_color_histogram(image):
    chans = cv2.split(image)
    hist = []
    for chan in chans:
        h = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist.append(h)
    return hist

def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def extract_lbp(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def plot_side_by_side(raw_img, clean_img, class_name):
    raw_hist = extract_color_histogram(raw_img)
    clean_hist = extract_color_histogram(clean_img)

    raw_edges = extract_edges(raw_img)
    clean_edges = extract_edges(clean_img)

    raw_lbp = extract_lbp(raw_img)
    clean_lbp = extract_lbp(clean_img)

    fig, axs = plt.subplots(4, 2, figsize=(14, 12))

    # Original Images
    axs[0, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(f"Raw Image - {class_name}")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title(f"Cleaned Image - {class_name}")
    axs[0, 1].axis('off')

    # Color Histograms
    colors = ['b', 'g', 'r']
    for i, color in enumerate(colors):
        axs[1, 0].plot(raw_hist[i], color=color)
        axs[1, 1].plot(clean_hist[i], color=color)
    axs[1, 0].set_xlim([0, 256])
    axs[1, 0].set_title('Raw Color Histogram')
    axs[1, 1].set_xlim([0, 256])
    axs[1, 1].set_title('Cleaned Color Histogram')

    # Edge Maps
    axs[2, 0].imshow(raw_edges, cmap='gray')
    axs[2, 0].set_title('Raw Edge Map')
    axs[2, 0].axis('off')

    axs[2, 1].imshow(clean_edges, cmap='gray')
    axs[2, 1].set_title('Cleaned Edge Map')
    axs[2, 1].axis('off')

    # LBP Histograms
    axs[3, 0].bar(range(len(raw_lbp)), raw_lbp)
    axs[3, 0].set_title('Raw LBP Histogram')
    axs[3, 1].bar(range(len(clean_lbp)), clean_lbp)
    axs[3, 1].set_title('Cleaned LBP Histogram')

    plt.tight_layout()
    plt.show()



import os
import random


raw_dataset_path = '/content/EuroSAT_RGB/EuroSAT_RGB'
cleaned_dataset_path = '/content/cleaned_dataset/EuroSAT_RGB'
class_name = 'AnnualCrop'  

raw_class_folder = os.path.join(raw_dataset_path, class_name)
clean_class_folder = os.path.join(cleaned_dataset_path, class_name)

raw_images = [f for f in os.listdir(raw_class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
clean_images = [f for f in os.listdir(clean_class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]


sample_img_name = random.choice(raw_images)
raw_img_path = os.path.join(raw_class_folder, sample_img_name)


if sample_img_name in clean_images:
    clean_img_path = os.path.join(clean_class_folder, sample_img_name)
else:
    clean_img_path = os.path.join(clean_class_folder, random.choice(clean_images))

raw_img = cv2.imread(raw_img_path)
clean_img = cv2.imread(clean_img_path)

plot_side_by_side(raw_img, clean_img, class_name)

#3.	Deep Learning Model 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # change based on your dataset classes count

# Data transforms (normalize and resize)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std, can adjust
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset from cleaned folder
dataset_path = '/content/cleaned_dataset/EuroSAT_RGB'
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Train-validation split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Define CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),  # downsampled by 2x2x2 pooling
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # final output layer without softmax
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate model, loss and optimizer
model = CustomCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and validation loop
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    train_loss = train_loss / train_total
    train_acc_history.append(train_acc)
    train_loss_history.append(train_loss)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / val_total
    val_acc_history.append(val_acc)
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()










