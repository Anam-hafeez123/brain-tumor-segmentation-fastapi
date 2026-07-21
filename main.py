import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from models.unet import UNet

app = FastAPI(title="Brain Tumor Segmentation API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(DEVICE)
model.load_state_dict(torch.load("best_unet_lgg.pth", map_location=DEVICE))
model.eval()

# Complete transformation pipeline matching LGG training parameters
transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform_pipeline(image)
    return tensor.unsqueeze(0).to(DEVICE)

@app.get("/")
def home():
    return {"status": "Active", "message": "API Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_tensor = preprocess_image(contents)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).squeeze().cpu()
        
        # Log probability stats
        min_p = probs.min().item()
        max_p = probs.max().item()
        mean_p = probs.mean().item()
        print(f"\n[DEBUG] Min: {min_p:.6f} | Max: {max_p:.6f} | Mean: {mean_p:.6f}")
        
        # Adaptive Thresholding: If overall probability is low, use percentile relative scaling
        if max_p > 0.1:
            mask = (probs > 0.5).float().numpy()
        else:
            # Normalize internal confidence range so relative mask is visible
            p_norm = (probs - min_p) / (max_p - min_p + 1e-8)
            mask = (p_norm > 0.6).float().numpy()

    mask_image = Image.fromarray((mask * 255).astype("uint8"))
    
    img_byte_arr = io.BytesIO()
    mask_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)