from flask import Flask, request, jsonify
import numpy as np
import os
from typing import Dict, Any

from src.audio.feature_extractor import AudioFeatureExtractor
from src.facial.feature_extractor import FacialFeatureExtractor
from src.models.depression_model import DepressionModel

app = Flask(__name__)

# Initialize feature extractors
audio_extractor = AudioFeatureExtractor()
facial_extractor = FacialFeatureExtractor()

# Load the trained model
model = DepressionModel.load("models/depression_model.h5")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    Expected JSON payload:
    {
        "audio_path": "path/to/audio.wav",
        "video_path": "path/to/video.mp4"
    }
    """
    try:
        data = request.get_json()
        
        # Extract features
        audio_features = audio_extractor.extract_features(data['audio_path'])
        facial_features = facial_extractor.extract_features(data['video_path'])
        
        # Combine features
        combined_features = np.concatenate([
            audio_features['mfccs'],
            audio_features['pitch'],
            [audio_features['energy']],
            [audio_features['zcr']],
            [audio_features['spectral_centroid']],
            facial_features['action_units']
        ])
        
        # Make prediction
        prediction = model.predict(combined_features.reshape(1, -1))
        
        return jsonify({
            "depression_probability": float(prediction),
            "is_depressed": bool(prediction > 0.5)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """
    Endpoint for extracting features without making predictions
    Expected JSON payload:
    {
        "audio_path": "path/to/audio.wav",
        "video_path": "path/to/video.mp4"
    }
    """
    try:
        data = request.get_json()
        
        # Extract features
        audio_features = audio_extractor.extract_features(data['audio_path'])
        facial_features = facial_extractor.extract_features(data['video_path'])
        
        return jsonify({
            "audio_features": {k: v.tolist() for k, v in audio_features.items()},
            "facial_features": {k: v.tolist() for k, v in facial_features.items()}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 