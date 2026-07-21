import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet

def run_prediction(img_path, mask_path):
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("best_unet_lgg.pth", map_location=DEVICE))
    model.eval()

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    input_tensor = cv2.resize(img, (256, 256)).astype("float32") / 255.0
    tensor = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).cpu().squeeze().numpy().astype(np.uint8)

    mask_resized = cv2.resize(mask, (256, 256)) > 0

    overlay = input_tensor.copy()
    overlay[pred_mask == 1] = [1.0, 0.0, 0.0]

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(input_tensor); ax[0].set_title("Input MRI")
    ax[1].imshow(mask_resized, cmap="gray"); ax[1].set_title("Ground Truth Mask")
    ax[2].imshow(pred_mask, cmap="gray"); ax[2].set_title("Prediction Mask")
    ax[3].imshow(overlay); ax[3].set_title("Overlay")

    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig("prediction_result.png")
    print(" Saved prediction plot to prediction_result.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <path_to_image> <path_to_mask>")
    else:
        run_prediction(sys.argv[1], sys.argv[2])