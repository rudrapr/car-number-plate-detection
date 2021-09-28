print(0)
from fastapi import FastAPI, UploadFile, File
print(1)
from plate_detector import Detector
print(2)
import shutil
print(3)

app = FastAPI()
print(4)
detector = Detector()
print(5)

@app.post('/detect_image')
async def detect_image(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    print('no error')
    try:
        result = detector.detect(file.filename)
        return {'bb_box': result.tolist()}
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        return {'error': message}


@app.get('/')
async def index():
    print(6)
    return {'msg': 'hello AI!'}
