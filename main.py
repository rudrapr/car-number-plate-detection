from fastapi import FastAPI, UploadFile, File
from plate_detector import Detector
import shutil
import numpy as np

app = FastAPI()
detector = Detector()


@app.post('/detect_image')
async def detect_image(file: UploadFile = File(...)):
    result = detector.detect(file)
    return {'bb_box': result.tolist()}
    # try:
    #     result = detector.detect(file)
    #     return {'bb_box': result.tolist()}
    # except Exception as ex:
    #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     return {'error': message}


@app.get('/')
async def index():
    print(6)
    return {'msg': 'hello AI!'}
