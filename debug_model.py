import os
from app.core.config import settings

print(f'Model path: {settings.MODEL_PATH}')
print(f'Path exists: {os.path.exists(settings.MODEL_PATH)}')
print('Attempting to load model...')

from ultralytics import YOLO
try:
    model = YOLO(settings.MODEL_PATH)
    print('Model loaded successfully')
    print(f'Model names: {model.names}')
    print(f'Model type: {type(model)}')
except Exception as e:
    print(f'Error loading model: {e}')
    import traceback
    traceback.print_exc()
