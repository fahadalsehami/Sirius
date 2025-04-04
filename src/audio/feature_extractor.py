import librosa
import numpy as np
from typing import Dict, Any, Tuple

class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract audio features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing various audio features
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches, axis=1)
        
        # Extract energy
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        return {
            'mfccs': mfccs_mean,
            'pitch': pitch_mean,
            'energy': energy_mean,
            'zcr': zcr_mean,
            'spectral_centroid': spectral_centroid_mean
        }

    def get_feature_dimension(self) -> int:
        """Return the total dimension of all features"""
        return 13 + 1 + 1 + 1 + 1  # mfccs(13) + pitch(1) + energy(1) + zcr(1) + spectral_centroid(1) 