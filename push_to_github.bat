@echo off
echo Pushing Depression Detection Project to GitHub...

:: Check if Git is installed
git --version > nul 2>&1
if errorlevel 1 (
    echo Git is not installed. Please install Git first.
    exit /b 1
)

:: Check if repository is already initialized
if not exist .git (
    echo Initializing Git repository...
    git init
    if errorlevel 1 (
        echo Failed to initialize Git repository.
        exit /b 1
    )
)

:: Add all files
echo Adding files to Git...
git add .
if errorlevel 1 (
    echo Failed to add files to Git.
    exit /b 1
)

:: Commit changes
echo Committing changes...
git commit -m "Initial commit: Depression Detection Project"
if errorlevel 1 (
    echo Failed to commit changes.
    exit /b 1
)

:: Set remote to sirius repository
echo Setting remote repository...
git remote add origin https://github.com/fahadalsehami/Sirius.git
if errorlevel 1 (
    echo Remote already exists. Updating URL...
    git remote set-url origin https://github.com/fahadalsehami/Sirius.git
)

:: Push to GitHub
echo Pushing to GitHub...
git push -u origin main
if errorlevel 1 (
    echo Failed to push to GitHub.
    echo If this is your first push, you might need to:
    echo 1. Check your GitHub credentials
    echo 2. Try pushing again
    exit /b 1
)

echo.
echo Successfully pushed to GitHub!
echo.
echo Press any key to exit...
pause > nul 