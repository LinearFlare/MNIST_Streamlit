import io
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model: Optional[MnistCNN] = None

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "models" / "mnist_cnn.pth"

TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    ),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if WEIGHTS_PATH.exists():
        model = MnistCNN().to(device)

        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location=device
        )

        model.load_state_dict(state_dict)
        model.eval()

    yield


app = FastAPI(
    title="MNIST CNN API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "device": str(device),
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "healthy": True,
        "model_ready": model is not None,
        "device": str(device),
    }


@app.get("/debug")
def debug():
    return {
        "cwd": str(Path.cwd()),
        "base_dir": str(BASE_DIR),
        "weights_path": str(WEIGHTS_PATH),
        "weights_exists": WEIGHTS_PATH.exists(),
        "device": str(device),
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        content = await file.read()

        image = Image.open(
            io.BytesIO(content)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    tensor = (
        TRANSFORM(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

        pred = probs.argmax().item()

        confidence = probs[pred].item()

    return JSONResponse({
        "prediction": pred,
        "confidence": round(
            confidence * 100,
            2
        ),
        "probabilities": {
            str(i): round(
                probs[i].item() * 100,
                2
            )
            for i in range(10)
        }
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )