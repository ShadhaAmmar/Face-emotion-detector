import os
import cv2
from camera import Camera

import sys

# paths
CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Pre-trained model not found at {MODEL_PATH}.")
        print("Download a Keras .h5 model trained on FER2013 or similar and place it as 'emotion_model.h5' in the project directory.")
        return
        
    # use command line argument as source if provided, otherwise default to webcam index 0
    source = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg
        
    cam = Camera(model_path=MODEL_PATH, cascade_path=CASCADE_PATH, source=source)
    cam.run()

if __name__ == "__main__":
    main()
