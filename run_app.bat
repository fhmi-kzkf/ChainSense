@echo off
echo.
echo ===============================================
echo          Starting ChainSense                  
echo     Supply Chain Risk Analyzer              
echo ===============================================
echo.

:: Check if virtual environment exists
if not exist chainsense_env\Scripts\activate.bat (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to install ChainSense.
    echo.
    echo Press any key to run setup now...
    pause >nul
    call setup.bat
    exit /b
)

echo ✓ Virtual environment found
echo ✓ Activating environment...
call chainsense_env\Scripts\activate.bat

echo ✓ Checking application files...
if not exist app.py (
    echo ERROR: app.py not found in current directory
    echo Please ensure you're running this from the ChainSense folder
    pause
    exit /b 1
)

echo ✓ Starting ChainSense application...
echo.
echo 🌐 ChainSense will open in your default web browser
echo 🔗 URL: http://localhost:8501
echo.
echo 📝 Application Log:
echo ----------------------------------------

:: Start Streamlit with custom configuration
streamlit run app.py --server.port 8501 --server.address localhost --theme.base light

echo.
echo ===============================================
echo        ChainSense Application Stopped        
echo ===============================================
echo.
echo Press any key to exit...
pause >nul