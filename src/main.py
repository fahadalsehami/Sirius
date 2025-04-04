import argparse
import os
from pathlib import Path
from src.data.download_dataset import DatasetDownloader
from src.data.preprocess_dataset import DatasetPreprocessor
from src.models.train_model import ModelTrainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Depression Detection Pipeline")
    
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory to store the dataset")
    
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to store the trained model")
    
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading the dataset")
    
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing the dataset")
    
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training the model")
    
    return parser.parse_args()

def main():
    """Run the depression detection pipeline"""
    args = parse_args()
    
    # Create directories
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    # Step 1: Download dataset
    if not args.skip_download:
        print("\n=== Step 1: Downloading Dataset ===")
        downloader = DatasetDownloader(data_dir=data_dir)
        downloader.prepare_dataset()
    else:
        print("\n=== Step 1: Skipping Dataset Download ===")
    
    # Step 2: Preprocess dataset
    if not args.skip_preprocess:
        print("\n=== Step 2: Preprocessing Dataset ===")
        preprocessor = DatasetPreprocessor(data_dir=data_dir)
        preprocessor.preprocess_dataset()
    else:
        print("\n=== Step 2: Skipping Dataset Preprocessing ===")
    
    # Step 3: Train model
    if not args.skip_train:
        print("\n=== Step 3: Training Model ===")
        trainer = ModelTrainer(data_dir=data_dir, model_dir=model_dir)
        trainer.train_and_evaluate()
    else:
        print("\n=== Step 3: Skipping Model Training ===")
    
    print("\n=== Pipeline Completed ===")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")

if __name__ == "__main__":
    main() 