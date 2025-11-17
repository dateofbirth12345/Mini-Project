"""
Train a binary classifier to distinguish between plastic and non-plastic debris.
This model will be used to classify detected debris regions.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

from utils import load_config


class DebrisClassifierDataset(Dataset):
    """Dataset for plastic/non-plastic classification."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # 0 for non-plastic, 1 for plastic
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a black image if loading fails
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_dataset_from_folders(plastic_dir: str, non_plastic_dir: str):
    """Load dataset from two folders: plastic/ and non-plastic/"""
    image_paths = []
    labels = []
    
    # Load plastic images
    if os.path.isdir(plastic_dir):
        for filename in os.listdir(plastic_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(plastic_dir, filename))
                labels.append(1)  # 1 = plastic
    
    # Load non-plastic images
    if os.path.isdir(non_plastic_dir):
        for filename in os.listdir(non_plastic_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(non_plastic_dir, filename))
                labels.append(0)  # 0 = non-plastic
    
    return image_paths, labels


def create_model(num_classes=2):
    """Create a ResNet-based classifier."""
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_classifier(
    plastic_dir: str,
    non_plastic_dir: str,
    output_dir: str = "models/classifier",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    device: str = "auto"
):
    """
    Train a binary classifier for plastic/non-plastic classification.
    
    Args:
        plastic_dir: Directory containing plastic images
        non_plastic_dir: Directory containing non-plastic images
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        val_split: Validation split ratio
        device: Device to use ('auto', 'cuda', or 'cpu')
    """
    print("Loading dataset...")
    image_paths, labels = load_dataset_from_folders(plastic_dir, non_plastic_dir)
    
    if len(image_paths) == 0:
        raise ValueError("No images found in the provided directories!")
    
    print(f"Found {sum(labels)} plastic images and {len(labels) - sum(labels)} non-plastic images")
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, random_state=42, stratify=labels
    )
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DebrisClassifierDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DebrisClassifierDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(output_dir, "plastic_classifier_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(output_dir, 'plastic_classifier_best.pt')}")
    
    # Save model info
    info = {
        'plastic_dir': plastic_dir,
        'non_plastic_dir': non_plastic_dir,
        'total_images': len(image_paths),
        'train_images': len(train_paths),
        'val_images': len(val_paths),
        'best_val_acc': float(best_val_acc),
        'epochs': epochs
    }
    with open(os.path.join(output_dir, "classifier_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    return model


def main():
    """Main function to train the classifier."""
    cfg = load_config("configs/config.yaml")
    
    # Get dataset paths from config or use defaults
    classifier_cfg = cfg.get("classifier", {})
    plastic_dir = classifier_cfg.get("plastic_dir", "data/classifier/plastic")
    non_plastic_dir = classifier_cfg.get("non_plastic_dir", "data/classifier/non_plastic")
    output_dir = classifier_cfg.get("output_dir", "models/classifier")
    epochs = classifier_cfg.get("epochs", 20)
    batch_size = classifier_cfg.get("batch_size", 32)
    learning_rate = classifier_cfg.get("learning_rate", 0.001)
    device = classifier_cfg.get("device", "auto")
    
    print("=" * 60)
    print("Plastic/Non-Plastic Classifier Training")
    print("=" * 60)
    print(f"Plastic images directory: {plastic_dir}")
    print(f"Non-plastic images directory: {non_plastic_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("=" * 60)
    
    train_classifier(
        plastic_dir=plastic_dir,
        non_plastic_dir=non_plastic_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )


if __name__ == "__main__":
    main()

