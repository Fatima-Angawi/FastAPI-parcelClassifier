from PIL import Image
from app.services.model_engine import predict

try:
    result = predict('test_image_improved.jpg')
    print(f'Prediction successful:')
    print(f'  Top class: {result.top_class}')
    print(f'  Top score: {result.top_score}')
    print(f'  All probs: {result.class_probabilities}')
except Exception as e:
    print(f'Error in predict: {e}')
    import traceback
    traceback.print_exc()
