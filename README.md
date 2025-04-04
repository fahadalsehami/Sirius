# Depression Detection using Multimodal Biomarkers

This project implements a machine learning system for detecting depression using voice and facial biomarkers. It combines acoustic features from voice recordings and facial expressions to provide a comprehensive analysis.

## Features

- Voice analysis using MFCCs and other acoustic features
- Facial expression analysis using Action Units (AUs)
- Multimodal fusion for improved accuracy
- REST API for easy integration
- Comprehensive evaluation metrics

## Prerequisites

- Python 3.9 or higher
- Git
- Visual Studio Code (recommended)
- CMake (for dlib installation)

## Setup

1. Create a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dlib facial landmark predictor:
```bash
# Create models directory
mkdir models

# Download the predictor file (you'll need to manually download this)
# Place shape_predictor_68_face_landmarks.dat in the models/ directory
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

4. Run the application:
```bash
python src/api/app.py
```

## Project Structure

```
sirius/
├── data/               # Data storage
├── models/            # Trained models
├── src/              # Source code
│   ├── audio/        # Audio processing
│   ├── facial/       # Facial analysis
│   ├── models/       # ML models
│   └── api/          # API endpoints
├── tests/            # Test files
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

## API Usage

The API provides endpoints for:
- Feature extraction from audio/video
- Depression prediction
- Model evaluation

Example API call:
```bash
# Windows
curl -X POST http://localhost:5000/predict ^
     -H "Content-Type: application/json" ^
     -d "{\"audio_path\": \"path/to/audio.wav\", \"video_path\": \"path/to/video.mp4\"}"

# Linux/Mac
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"audio_path\": \"path/to/audio.wav\", \"video_path\": \"path/to/video.mp4\"}"
```

## Development

1. Run tests:
```bash
pytest tests/
```

2. Check code style:
```bash
flake8 .
```

## Pushing to GitHub

The project is already connected to the GitHub repository: [Sirius](https://github.com/fahadalsehami/Sirius)

### Windows
```bash
.\push_to_github.bat
```

### Linux/Mac
```bash
chmod +x push_to_github.sh
./push_to_github.sh
```

## Troubleshooting

1. If you encounter issues installing dlib:
   - Make sure you have CMake installed
   - On Windows, you might need to install Visual Studio Build Tools
   - Try installing dlib separately: `pip install dlib`

2. If you get CUDA errors with TensorFlow:
   - Install the CPU-only version: `pip install tensorflow-cpu`

3. If you have issues pushing to GitHub:
   - Check your GitHub credentials
   - Try setting the remote URL manually: `git remote set-url origin https://github.com/fahadalsehami/Sirius.git`

## License

MIT License 