import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Dict, Any

class DepressionModel:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build the neural network architecture"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # First dense layer with batch normalization
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Second dense layer
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        return history.history
    
    def predict(self, features: np.ndarray) -> float:
        """
        Make predictions on new data
        
        Args:
            features: Input features
            
        Returns:
            Probability of depression
        """
        return float(self.model.predict(features)[0][0])
    
    def save(self, path: str):
        """Save the model to disk"""
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'DepressionModel':
        """Load a saved model from disk"""
        model = cls(input_dim=17)  # 13 (MFCCs) + 4 (facial features)
        model.model = tf.keras.models.load_model(path)
        return model 