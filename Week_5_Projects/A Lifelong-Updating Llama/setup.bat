@echo off
REM SEAL-drip Setup Script for Windows
REM Simple script to set up the environment and test data collection

echo 🚀 SEAL-drip: A Lifelong-Updating Llama Setup
echo ==============================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo ✅ Found Python
python --version

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip not found. Please install pip and try again.
    pause
    exit /b 1
)

echo ✅ Found pip
pip --version

REM Install dependencies
echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

REM Run tests
echo.
echo 🧪 Running tests to verify setup...
python test_collection.py

if %errorlevel% equ 0 (
    echo.
    echo 🎉 Setup complete! Your SEAL-drip environment is ready.
    echo.
    echo Next steps:
    echo   - Run 'python collect.py' to start collecting data
    echo   - Check the data/ directory for collected files
    echo   - Edit config.py to customize collection parameters
    echo.
) else (
    echo.
    echo ⚠️  Setup completed but tests had issues.
    echo This might be normal if some APIs are temporarily unavailable.
    echo Try running 'python collect.py' to test data collection manually.
    echo.
)

pause 