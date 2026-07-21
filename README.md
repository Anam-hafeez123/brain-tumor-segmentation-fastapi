# 🧠 Brain Tumor Segmentation API (U-Net & FastAPI)

An end-to-end deep learning web API built with PyTorch and FastAPI to segment low-grade gliomas (LGG) from brain MRI scans using a U-Net architecture.

---

## 🚀 Features

- **Deep Learning Model:** Implements a U-Net CNN model trained on brain MRI segmentation datasets.
- **RESTful API:** Powered by FastAPI for fast, scalable image inference.
- **Interactive UI:** Built-in Swagger UI documentation for uploading MRI scans (`.tif`, `.png`, `.jpg`) and previewing predicted segmentation masks in real-time.
- **Adaptive Post-Processing:** Normalizes and rescales raw confidence logits for accurate feature visualization.

---

## 🛠️ Tech Stack

- **Framework:** FastAPI / Uvicorn
- **Deep Learning:** PyTorch, torchvision
- **Data & Image Processing:** OpenCV, Pillow, NumPy
- **Language:** Python 3.10+

---

## 📂 Project Structure

```text
unet-medical/
├── models/
│   └── unet.py             # U-Net model architecture class definition
├── best_unet_lgg.pth       # Saved PyTorch trained weights (gitignored)
├── main.py                 # FastAPI backend server & prediction endpoints
├── .gitignore              # Ignored files (virtual environments, weights, raw data)
└── README.md               # Project documentation
