from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from typing import List
from PIL import Image
import base64
import io
import torchvision.transforms as transforms

app = FastAPI()

# Load the ONNX model
onnx_session = ort.InferenceSession("model_onnx.onnx")

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
        image_tensor = transform(image).unsqueeze(0)
        image_np = image_tensor.numpy()

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input format") from e

    inputs = {onnx_session.get_inputs()[0].name: image_np}

    outputs = onnx_session.run(None, inputs)
    predicted_labels = np.argmax(outputs[0], axis=1)

    predictions = predicted_labels.tolist()
    return PredictionResponse(predictions=predictions)
