#!/bin/bash

echo "Pushing Depression Detection Project to GitHub..."

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install Git first."
    exit 1
fi

# Check if repository is already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    if [ $? -ne 0 ]; then
        echo "Failed to initialize Git repository."
        exit 1
    fi
fi

# Add all files
echo "Adding files to Git..."
git add .
if [ $? -ne 0 ]; then
    echo "Failed to add files to Git."
    exit 1
fi

# Commit changes
echo "Committing changes..."
git commit -m "Initial commit: Depression Detection Project"
if [ $? -ne 0 ]; then
    echo "Failed to commit changes."
    exit 1
fi

# Set remote to sirius repository
echo "Setting remote repository..."
git remote add origin https://github.com/fahadalsehami/Sirius.git || git remote set-url origin https://github.com/fahadalsehami/Sirius.git

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main
if [ $? -ne 0 ]; then
    echo "Failed to push to GitHub."
    echo "If this is your first push, you might need to:"
    echo "1. Check your GitHub credentials"
    echo "2. Try pushing again"
    exit 1
fi

echo ""
echo "Successfully pushed to GitHub!" 