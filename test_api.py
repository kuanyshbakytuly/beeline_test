import base64
import requests

with open("image copy.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

payload = {"image_base64": base64_image}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())