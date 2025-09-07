@echo off
echo.
echo ===============================================
echo        ChainSense Setup & Installation        
echo        Supply Chain Risk Analyzer            
echo ===============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo.
    pause
    exit /b 1
)

echo ✓ Python found: 
for /f "tokens=*" %%i in ('python --version') do echo   %%i
echo.

echo [1/5] Creating Python virtual environment...
if exist chainsense_env (
    echo   Virtual environment already exists, skipping...
) else (
    python -m venv chainsense_env
    if errorlevel 1 (
        echo   ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo   ✓ Virtual environment created successfully
)
echo.

echo [2/5] Activating virtual environment...
call chainsense_env\Scripts\activate.bat
if errorlevel 1 (
    echo   ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo   ✓ Virtual environment activated
echo.

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo   ✓ Pip upgraded to latest version
echo.

echo [4/5] Installing ChainSense dependencies...
echo   This may take a few minutes...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo   ERROR: Failed to install dependencies
    echo   Please check requirements.txt and try again
    pause
    exit /b 1
)
echo   ✓ All dependencies installed successfully
echo.

echo [5/5] Verifying installation...
python -c "import streamlit, pandas, networkx, pyvis, plotly; print('✓ All core modules imported successfully')"
if errorlevel 1 (
    echo   WARNING: Some modules may not be properly installed
    echo   The application might still work with fallback options
)
echo.

echo ===============================================
echo             Installation Complete!            
echo ===============================================
echo.
echo 🚀 To start ChainSense:
echo    1. Run: run_app.bat
echo    2. Or manually: chainsense_env\Scripts\activate.bat && streamlit run app.py
echo.
echo 📚 For help and documentation:
echo    - README.md: User guide and features
echo    - DEVELOPMENT.md: Developer documentation
echo    - CHANGELOG.md: Version history
echo.
echo 🌐 The application will open in your default web browser
echo    URL: http://localhost:8501
echo.
echo Press any key to continue...
pause >nul
echo.
echo Would you like to start ChainSense now? (Y/N)
set /p choice="Enter your choice: "
if /i "%choice%"=="Y" (
    echo.
    echo Starting ChainSense...
    call run_app.bat
) else (
    echo.
    echo Setup complete. Run 'run_app.bat' when ready to start.
)