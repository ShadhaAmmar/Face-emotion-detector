import os
import cv2
from camera import Camera

import sys

# paths
CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
JSON_PATH = os.path.join(os.path.dirname(__file__), 'emotion_model.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')

def main():
    # check for json config first, then direct h5 model
    if os.path.exists(JSON_PATH):
        model_path = JSON_PATH
    elif os.path.exists(MODEL_PATH):
        model_path = MODEL_PATH
    else:
        print("Error: Pre-trained model files not found.")
        print("Please place 'emotion_model.json' + 'emotion_model.h5' or just 'emotion_model.h5' in the project directory.")
        return
        
    # use command line argument as source if provided, otherwise default to webcam index 0
    source = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg
        
    cam = Camera(model_path=model_path, cascade_path=CASCADE_PATH, source=source)
    cam.run()

if __name__ == "__main__":
    main()
