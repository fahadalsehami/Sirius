import librosa
import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

class ClassificationLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

@dataclass
class AudioFeature:
    value: float
    unit: str
    classification: ClassificationLevel
    confidence: float

class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
        # Initialize feature thresholds
        self.thresholds = {
            'energy': {
                'low': 40,
                'high': 70,
                'unit': 'dB'
            },
            'zcr': {
                'low': 500,
                'high': 1500,
                'unit': 'Hz'
            },
            'autocorrelation': {
                'low': 0.3,
                'high': 0.7,
                'unit': 'coefficient'
            },
            'spectral_centroid': {
                'low': 2000,
                'high': 6000,
                'unit': 'Hz'
            },
            'spectral_bandwidth': {
                'low': 1000,
                'high': 3000,
                'unit': 'Hz'
            },
            'spectral_rolloff': {
                'low': 2000,
                'high': 6000,
                'unit': 'Hz'
            },
            'spectral_flatness': {
                'low': 0.1,
                'high': 0.5,
                'unit': 'ratio'
            },
            'pitch': {
                'low': 100,
                'high': 300,
                'unit': 'Hz'
            },
            'formant_f1': {
                'low': 300,
                'high': 900,
                'unit': 'Hz'
            },
            'formant_f2': {
                'low': 850,
                'high': 2500,
                'unit': 'Hz'
            },
            'formant_f3': {
                'low': 2300,
                'high': 3500,
                'unit': 'Hz'
            },
            'hnr': {
                'low': 20,
                'high': 60,
                'unit': 'dB'
            },
            'jitter': {
                'low': 0.5,
                'high': 1.5,
                'unit': '%'
            },
            'shimmer': {
                'low': 1.0,
                'high': 3.0,
                'unit': '%'
            },
            'vot': {
                'low': 20,
                'high': 80,
                'unit': 'ms'
            }
        }

    def _classify_feature(self, feature_name: str, value: float) -> ClassificationLevel:
        """Classify a feature value based on its thresholds"""
        thresholds = self.thresholds[feature_name]
        if value < thresholds['low']:
            return ClassificationLevel.LOW
        elif value > thresholds['high']:
            return ClassificationLevel.HIGH
        return ClassificationLevel.MEDIUM

    def _calculate_confidence(self, feature_name: str, value: float) -> float:
        """Calculate confidence level for a feature measurement"""
        base_confidence = 85.0
        
        # Adjust confidence based on feature-specific factors
        if feature_name in ['pitch', 'formant_f1', 'formant_f2', 'formant_f3']:
            # Higher confidence for frequencies within typical human range
            if self.thresholds[feature_name]['low'] <= value <= self.thresholds[feature_name]['high']:
                return min(100, base_confidence + 10)
            return max(0, base_confidence - 20)
        
        elif feature_name in ['jitter', 'shimmer']:
            # Higher confidence for normal voice quality measures
            if value < self.thresholds[feature_name]['high']:
                return min(100, base_confidence + 5)
            return max(0, base_confidence - 15)
        
        return base_confidence

    def extract_time_domain_features(self, y: np.ndarray, sr: int) -> Dict[str, AudioFeature]:
        """Extract time-domain features"""
        # Energy (RMS)
        energy = librosa.feature.rms(y=y)[0]
        energy_db = 20 * np.log10(np.mean(energy) + 1e-10)
        
        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr) * sr
        
        # Autocorrelation
        autocorr = librosa.autocorrelate(y)
        autocorr_max = np.max(autocorr)
        
        return {
            'energy': AudioFeature(
                value=energy_db,
                unit=self.thresholds['energy']['unit'],
                classification=self._classify_feature('energy', energy_db),
                confidence=self._calculate_confidence('energy', energy_db)
            ),
            'zcr': AudioFeature(
                value=zcr_mean,
                unit=self.thresholds['zcr']['unit'],
                classification=self._classify_feature('zcr', zcr_mean),
                confidence=self._calculate_confidence('zcr', zcr_mean)
            ),
            'autocorrelation': AudioFeature(
                value=autocorr_max,
                unit=self.thresholds['autocorrelation']['unit'],
                classification=self._classify_feature('autocorrelation', autocorr_max),
                confidence=self._calculate_confidence('autocorrelation', autocorr_max)
            )
        }

    def extract_frequency_domain_features(self, y: np.ndarray, sr: int) -> Dict[str, AudioFeature]:
        """Extract frequency-domain features"""
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroid)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        bandwidth_mean = np.mean(spectral_bandwidth)
        
        # Spectral Roll-off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        rolloff_mean = np.mean(spectral_rolloff)
        
        # Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_mean = np.mean(spectral_flatness)
        
        return {
            'spectral_centroid': AudioFeature(
                value=centroid_mean,
                unit=self.thresholds['spectral_centroid']['unit'],
                classification=self._classify_feature('spectral_centroid', centroid_mean),
                confidence=self._calculate_confidence('spectral_centroid', centroid_mean)
            ),
            'spectral_bandwidth': AudioFeature(
                value=bandwidth_mean,
                unit=self.thresholds['spectral_bandwidth']['unit'],
                classification=self._classify_feature('spectral_bandwidth', bandwidth_mean),
                confidence=self._calculate_confidence('spectral_bandwidth', bandwidth_mean)
            ),
            'spectral_rolloff': AudioFeature(
                value=rolloff_mean,
                unit=self.thresholds['spectral_rolloff']['unit'],
                classification=self._classify_feature('spectral_rolloff', rolloff_mean),
                confidence=self._calculate_confidence('spectral_rolloff', rolloff_mean)
            ),
            'spectral_flatness': AudioFeature(
                value=flatness_mean,
                unit=self.thresholds['spectral_flatness']['unit'],
                classification=self._classify_feature('spectral_flatness', flatness_mean),
                confidence=self._calculate_confidence('spectral_flatness', flatness_mean)
            )
        }

    def extract_perceptual_features(self, y: np.ndarray, sr: int) -> Dict[str, AudioFeature]:
        """Extract perceptual features"""
        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/10])
        
        # Formants (simplified estimation)
        formants = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
        formant_f1 = np.percentile(formants, 25)
        formant_f2 = np.percentile(formants, 50)
        formant_f3 = np.percentile(formants, 75)
        
        # HNR (simplified estimation)
        hnr = librosa.effects.preemphasis(y)
        hnr_db = 20 * np.log10(np.mean(np.abs(hnr)) + 1e-10)
        
        # Jitter and Shimmer (simplified estimation)
        jitter = np.std(pitch_mean) / np.mean(pitch_mean) * 100
        shimmer = np.std(np.abs(y)) / np.mean(np.abs(y)) * 100
        
        # VOT (simplified estimation)
        vot = librosa.feature.zero_crossing_rate(y)[0]
        vot_ms = np.mean(vot) * 1000 / sr
        
        return {
            'pitch': AudioFeature(
                value=pitch_mean,
                unit=self.thresholds['pitch']['unit'],
                classification=self._classify_feature('pitch', pitch_mean),
                confidence=self._calculate_confidence('pitch', pitch_mean)
            ),
            'formant_f1': AudioFeature(
                value=formant_f1,
                unit=self.thresholds['formant_f1']['unit'],
                classification=self._classify_feature('formant_f1', formant_f1),
                confidence=self._calculate_confidence('formant_f1', formant_f1)
            ),
            'formant_f2': AudioFeature(
                value=formant_f2,
                unit=self.thresholds['formant_f2']['unit'],
                classification=self._classify_feature('formant_f2', formant_f2),
                confidence=self._calculate_confidence('formant_f2', formant_f2)
            ),
            'formant_f3': AudioFeature(
                value=formant_f3,
                unit=self.thresholds['formant_f3']['unit'],
                classification=self._classify_feature('formant_f3', formant_f3),
                confidence=self._calculate_confidence('formant_f3', formant_f3)
            ),
            'hnr': AudioFeature(
                value=hnr_db,
                unit=self.thresholds['hnr']['unit'],
                classification=self._classify_feature('hnr', hnr_db),
                confidence=self._calculate_confidence('hnr', hnr_db)
            ),
            'jitter': AudioFeature(
                value=jitter,
                unit=self.thresholds['jitter']['unit'],
                classification=self._classify_feature('jitter', jitter),
                confidence=self._calculate_confidence('jitter', jitter)
            ),
            'shimmer': AudioFeature(
                value=shimmer,
                unit=self.thresholds['shimmer']['unit'],
                classification=self._classify_feature('shimmer', shimmer),
                confidence=self._calculate_confidence('shimmer', shimmer)
            ),
            'vot': AudioFeature(
                value=vot_ms,
                unit=self.thresholds['vot']['unit'],
                classification=self._classify_feature('vot', vot_ms),
                confidence=self._calculate_confidence('vot', vot_ms)
            )
        }

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
        
        # Extract features from different domains
        time_domain_features = self.extract_time_domain_features(y, sr)
        freq_domain_features = self.extract_frequency_domain_features(y, sr)
        perceptual_features = self.extract_perceptual_features(y, sr)
        
        # Combine all features
        all_features = {**time_domain_features, **freq_domain_features, **perceptual_features}
        
        # Convert to numerical array
        feature_values = np.array([f.value for f in all_features.values()])
        
        return {
            'audio_features': feature_values,
            'feature_names': list(all_features.keys()),
            'feature_units': [f.unit for f in all_features.values()],
            'feature_classifications': [f.classification.value for f in all_features.values()],
            'feature_confidence': [f.confidence for f in all_features.values()]
        }

    def get_feature_dimension(self) -> int:
        """Return the total dimension of all features"""
        return len(self.thresholds)  # Number of features 