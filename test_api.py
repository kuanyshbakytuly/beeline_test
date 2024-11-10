import base64
import requests

img_path = "0.png"
with open(img_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

payload = {"image_base64": base64_image}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())