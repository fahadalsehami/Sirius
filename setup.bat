@echo off
echo Setting up Depression Detection Project...

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.9 or higher.
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

:: Create models directory
echo Creating models directory...
if not exist models mkdir models

echo.
echo Setup completed successfully!
echo.
echo Next steps:
echo 1. Download shape_predictor_68_face_landmarks.dat from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
echo 2. Extract and place the .dat file in the models/ directory
echo 3. Run the application with: python src/api/app.py
echo.
echo Press any key to exit...
pause > nul 