import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.models.depression_model import DepressionModel

class ModelTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        """Initialize the model trainer"""
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model parameters
        self.input_dim = 17  # 13 (MFCCs) + 4 (facial features)
        self.epochs = 50
        self.batch_size = 32
    
    def load_data(self):
        """Load the preprocessed data"""
        print("Loading preprocessed data...")
        
        X_train = np.load(self.features_dir / "X_train.npy")
        X_val = np.load(self.features_dir / "X_val.npy")
        X_test = np.load(self.features_dir / "X_test.npy")
        y_train = np.load(self.features_dir / "y_train.npy")
        y_val = np.load(self.features_dir / "y_val.npy")
        y_test = np.load(self.features_dir / "y_test.npy")
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the depression detection model"""
        print("Training model...")
        
        # Initialize model
        model = DepressionModel(input_dim=self.input_dim)
        
        # Train model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        
        # Save model
        model.save(self.model_dir / "depression_model.h5")
        print(f"Model saved to {self.model_dir / 'depression_model.h5'}")
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model on the test set"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred_prob = model.predict(X_test.reshape(-1, self.input_dim))
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.model_dir / "roc_curve.png")
        print(f"ROC curve saved to {self.model_dir / 'roc_curve.png'}")
        
        return {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": cm,
            "roc_auc": roc_auc
        }
    
    def plot_training_history(self, history):
        """Plot the training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png")
        print(f"Training history saved to {self.model_dir / 'training_history.png'}")
    
    def train_and_evaluate(self):
        """Train and evaluate the model"""
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Train model
        model, history = self.train_model(X_train, y_train, X_val, y_val)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, X_test, y_test)
        
        print("Model training and evaluation completed!")
        return model, evaluation_results

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate() 