from fastapi import FastAPI, UploadFile, File

from src.ocr import OCR
from src.plate_detector import Detector

app = FastAPI()
detector = Detector()
OCR = OCR()


@app.post('/detect_image')
async def detect_image(file: UploadFile = File(...)):
    result, ROI = detector.detect(file)
    txt = OCR.recognize(ROI)
    return {'bb_box': result.tolist(), 'prediction': txt}
    # try:
    #     result = detector.detect(file)
    #     return {'bb_box': result.tolist()}
    # except Exception as ex:
    #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     return {'error': message}


@app.get('/')
async def index():
    return {'msg': 'hello AI!'}
