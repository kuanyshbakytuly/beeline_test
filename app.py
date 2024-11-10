from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import Model
from typing import List
from PIL import Image
import base64
import io
import torchvision.transforms as transforms

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(num_classes=4).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PredictionRequest(BaseModel):
    image_base64: str

class PredictionResponse(BaseModel):
    predictions: List[int]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input format") from e

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_labels = torch.max(outputs, dim=1)

    predictions = predicted_labels.cpu().tolist()
    return PredictionResponse(predictions=predictions)
