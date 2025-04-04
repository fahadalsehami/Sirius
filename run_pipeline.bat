@echo off
echo Running Depression Detection Pipeline...

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.9 or higher.
    exit /b 1
)

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
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

:: Run the pipeline
echo Running the pipeline...
python src/main.py %*
if errorlevel 1 (
    echo Failed to run the pipeline.
    exit /b 1
)

echo.
echo Pipeline completed successfully!
echo.
echo Press any key to exit...
pause > nul 