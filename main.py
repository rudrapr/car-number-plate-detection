from typing import Optional
from fastapi import FastAPI, UploadFile, File
from plate_detector import Detector
import shutil

app = FastAPI()

detector = Detector()


@app.post('/detectImage')
async def detect_image(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detector.detect(file.filename)
    return {'bb_box': result.tolist()}


@app.get('/')
async def index():
    return {}
