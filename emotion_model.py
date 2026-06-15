import numpy as np
from tf_keras.models import load_model
import cv2

class EmotionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

    def preprocess(self, face_img):
        # resize to 48x48, normalize and expand dimensions for the model
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict(self, face_img):
        processed = self.preprocess(face_img)
        preds = self.model.predict(processed, verbose=0)[0]
        
        # get indices sorted by probability descending
        sorted_indices = np.argsort(preds)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]
        
        top1_label = self.emotion_labels[top1_idx]
        top2_label = self.emotion_labels[top2_idx]
        
        top1_prob = preds[top1_idx]
        top2_prob = preds[top2_idx]
        
        # define complex emotions when top two classes have high probabilities
        if top1_prob > 0.35 and top2_prob > 0.20:
            pair = {top1_label, top2_label}
            if pair == {"Happy", "Surprised"}:
                return "Happily Surprised", top1_prob
            elif pair == {"Fearful", "Surprised"}:
                return "Awestruck", top1_prob
            elif pair == {"Angry", "Surprised"}:
                return "Outraged", top1_prob
            elif pair == {"Sad", "Surprised"}:
                return "Disappointed", top1_prob
            elif pair == {"Angry", "Disgusted"}:
                return "Hostile", top1_prob
            elif pair == {"Angry", "Sad"}:
                return "Frustrated", top1_prob
            elif pair == {"Fearful", "Sad"}:
                return "Anxious", top1_prob
            elif pair == {"Disgusted", "Sad"}:
                return "Remorseful", top1_prob
            elif pair == {"Happy", "Disgusted"}:
                return "Cynical", top1_prob
            elif pair == {"Neutral", "Sad"}:
                return "Melancholic", top1_prob
            elif pair == {"Neutral", "Happy"}:
                return "Content", top1_prob
            elif pair == {"Neutral", "Angry"}:
                return "Stern", top1_prob
                
        return top1_label, top1_prob
