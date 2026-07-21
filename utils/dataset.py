import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(128, 128)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize both to 128x128 for 4x faster CPU training
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size)

        # Convert to Tensor & Normalize Image
        image = TF.to_tensor(image)
        # Standard ImageNet normalization for better convergence
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert mask to Tensor and binarize (0.0 or 1.0)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask

def get_dataloaders(data_dir, batch_size=16, train_split=0.8):
    # Find all images and corresponding masks in Kaggle LGG dataset
    all_images = glob.glob(os.path.join(data_dir, "**/*[0-9].tif"), recursive=True)
    all_masks = [img.replace(".tif", "_mask.tif") for img in all_images]

    # Filter out missing masks if any
    valid_pairs = [(img, mask) for img, mask in zip(all_images, all_masks) if os.path.exists(mask)]
    image_paths, mask_paths = zip(*valid_pairs)

    full_dataset = BrainMRIDataset(image_paths, mask_paths)

    # Calculate train/validation split sizes
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders (num_workers=0 to prevent Windows thread locks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader