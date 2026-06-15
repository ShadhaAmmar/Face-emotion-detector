import numpy as np
from tf_keras.models import load_model
import cv2

class EmotionModel:
    def __init__(self, model_path):
        import os
        if model_path.endswith('.json'):
            with open(model_path, 'r') as f:
                model_json = f.read()
            from tf_keras.models import model_from_json
            self.model = model_from_json(model_json)
            weights_path = os.path.splitext(model_path)[0] + '.h5'
            self.model.load_weights(weights_path)
        else:
            self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

    def preprocess(self, face_img):
        # resize to 48x48, normalize and expand dimensions for the model
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
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
        
        # 1. high-intensity basic emotions (super-emotions)
        if top1_prob > 0.70:
            if top1_label == "Angry":
                return "Enraged", top1_prob
            elif top1_label == "Surprised":
                return "Shocked", top1_prob
            elif top1_label == "Fearful":
                return "Terrified", top1_prob
            elif top1_label == "Happy":
                return "Elated", top1_prob
            elif top1_label == "Sad":
                return "Devastated", top1_prob
        
        # 2. define complex compound emotions when top two classes have high probabilities
        if top1_prob > 0.35 and top2_prob > 0.20:
            pair = {top1_label, top2_label}
            if pair == {"Happy", "Surprised"}:
                return "Amazed", top1_prob
            elif pair == {"Fearful", "Surprised"}:
                return "Awestruck", top1_prob
            elif pair == {"Angry", "Surprised"}:
                return "Outraged", top1_prob
            elif pair == {"Sad", "Surprised"}:
                return "Disappointed", top1_prob
            elif pair == {"Angry", "Disgusted"}:
                return "Scary", top1_prob
            elif pair == {"Angry", "Sad"}:
                return "Frustrated", top1_prob
            elif pair == {"Fearful", "Sad"}:
                return "Anxious", top1_prob
            elif pair == {"Disgusted", "Sad"}:
                return "Remorseful", top1_prob
            elif pair == {"Happy", "Disgusted"}:
                return "Cynical", top1_prob
            elif pair == {"Neutral", "Sad"}:
                return "Shy", top1_prob
            elif pair == {"Neutral", "Happy"}:
                return "Flirty", top1_prob
            elif pair == {"Neutral", "Angry"}:
                return "Grumpy", top1_prob
            elif pair == {"Neutral", "Fearful"}:
                return "Shy", top1_prob
                
        return top1_label, top1_prob
