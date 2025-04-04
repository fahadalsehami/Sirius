import pytest
import numpy as np
from src.models.depression_model import DepressionModel
from src.audio.feature_extractor import AudioFeatureExtractor
from src.facial.feature_extractor import FacialFeatureExtractor

def test_model_initialization():
    """Test model initialization"""
    model = DepressionModel(input_dim=17)
    assert model.input_dim == 17
    assert model.model is not None

def test_model_prediction():
    """Test model prediction"""
    model = DepressionModel(input_dim=17)
    # Create dummy features
    features = np.random.rand(1, 17)
    prediction = model.predict(features)
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 1

def test_audio_feature_extractor():
    """Test audio feature extraction"""
    extractor = AudioFeatureExtractor()
    # Create a dummy audio file path
    # In a real test, you would use a real audio file
    with pytest.raises(FileNotFoundError):
        extractor.extract_features("dummy.wav")

def test_facial_feature_extractor():
    """Test facial feature extraction"""
    extractor = FacialFeatureExtractor()
    # Create a dummy video file path
    # In a real test, you would use a real video file
    with pytest.raises(FileNotFoundError):
        extractor.extract_features("dummy.mp4")

def test_feature_dimensions():
    """Test feature dimensions match model input"""
    audio_extractor = AudioFeatureExtractor()
    facial_extractor = FacialFeatureExtractor()
    
    total_dim = audio_extractor.get_feature_dimension() + facial_extractor.get_feature_dimension()
    assert total_dim == 17  # Should match model input dimension 