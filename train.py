import torch
import sys
import matplotlib.pyplot as plt

# Prevents CPU overload/throttling on Intel i3
torch.set_num_threads(2)

from torch.utils.data import random_split, DataLoader
from models.unet import UNet
from utils.dataset import get_dataloaders
from utils.metrics import BCEDiceLoss, calculate_metrics

def train():
    # Detect device (CUDA GPU, Apple Silicon MPS, or CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    EPOCHS = 50        # Set to 50 Epochs
    BATCH_SIZE = 16
    DATA_DIR = "./data/kaggle_3m"

    print("Step 1/3: Initializing dataset...", flush=True)
    full_train_loader, full_val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # -------------------------------------------------------------
    # CALCULATE SUBSET TO FORCE EXACTLY 20 BATCHES PER EPOCH
    # 20 batches * 16 batch_size = 320 training images needed
    # -------------------------------------------------------------
    target_train_images = 20 * BATCH_SIZE
    train_size = min(target_train_images, len(full_train_loader.dataset))

    train_subset, _ = random_split(
        full_train_loader.dataset, 
        [train_size, len(full_train_loader.dataset) - train_size]
    )

    # Scale validation set down proportionally (~10 validation batches)
    val_size = min(10 * BATCH_SIZE, len(full_val_loader.dataset))
    val_subset, _ = random_split(
        full_val_loader.dataset, 
        [val_size, len(full_val_loader.dataset) - val_size]
    )

    # Re-create DataLoaders with num_workers=0 to prevent Windows thread locks
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Step 2/3: Dataset ready! Using {len(train_loader)} batches per epoch.")

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss()

    best_dice = 0.0
    train_losses = []
    val_dices = []

    print(f"Step 3/3: Training started on device [{DEVICE.upper()}] for {EPOCHS} epochs...\n", flush=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Live batch progress bar
            sys.stdout.write(f"\rEpoch [{epoch+1:02d}/{EPOCHS:02d}] Batch {step+1}/{len(train_loader)} | Current Loss: {loss.item():.4f}")
            sys.stdout.flush()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                dice, _ = calculate_metrics(outputs, masks)
                val_dice += dice

        epoch_dice = val_dice / len(val_loader)
        val_dices.append(epoch_dice)

        print(f"\n--- Epoch {epoch+1:02d}/{EPOCHS:02d} Complete | Avg Loss: {epoch_loss:.4f} | Val Dice: {epoch_dice:.4f} ---", flush=True)

        # Save checkpoint if score improves
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(model.state_dict(), "best_unet_lgg.pth")
            print(f" --> Saved best model checkpoint (Val Dice: {epoch_dice:.4f})\n", flush=True)

    # Plot performance curves automatically when 50 epochs complete
    print("\nTraining complete! Generating performance graphs...")
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="#d9534f", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()

    # Validation Dice score curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), val_dices, label="Validation Dice", color="#5cb85c", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient Score")
    plt.title("Validation Dice Score Curve")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Graphs saved locally to 'training_results.png'!")
    plt.show()

if __name__ == "__main__":
    train()