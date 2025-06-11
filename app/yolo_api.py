from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import numpy as np
import cv2
import io
import os
import time

app = FastAPI()
MODEL_PATH = "runs/detect/best_model.pt"
MAX_WAIT_TIME = 600  # Tối đa 10 phút chờ (tùy chỉnh theo thời gian training)
WAIT_INTERVAL = 10   # Kiểm tra mỗi 10 giây

# Chờ file mô hình sẵn sàng
def wait_for_model():
    start_time = time.time()
    while not os.path.exists(MODEL_PATH):
        if time.time() - start_time > MAX_WAIT_TIME:
            raise Exception(f"Timeout waiting for {MODEL_PATH} after {MAX_WAIT_TIME} seconds")
        print(f"Waiting for {MODEL_PATH}... Retrying in {WAIT_INTERVAL}s")
        time.sleep(WAIT_INTERVAL)
    print(f"Model file {MODEL_PATH} is ready!")

wait_for_model()
model = YOLO(MODEL_PATH)

app.mount("/static", StaticFiles(directory="static"), name="static")

def draw_boxes(image: np.ndarray, results) -> np.ndarray:
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return image

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    results = model.predict(image, save=False)[0]
    image_with_boxes = draw_boxes(image, results)

    _, img_encoded = cv2.imencode(".jpg", image_with_boxes)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
