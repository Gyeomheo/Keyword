@echo off
echo ========================================================
echo       Samsung Keyword Engine - Automated Pipeline
echo ========================================================

:: 1. Check/Train Model
if exist "C:\Keyword\model\rf_classifier.joblib" (
    echo [Info] Found existing RF model.
    set /p REBUILD="Do you want to retrain the model? (y/n): "
) else (
    echo [Info] No model found. Training required.
    set REBUILD=y
)

if /i "%REBUILD%"=="y" (
    echo [Step 1] Training Random Forest Model (SBERT Vectors)...
    python train_rf.py
    if %ERRORLEVEL% NEQ 0 (
        echo [Error] Training failed!
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo [Step 1] Skipping training. Using existing model.
)

:: 2. Run Main Engine
echo.
echo [Step 2] Running Keyword Classification Engine...
python main.py

echo.
echo ========================================================
echo                 Pipeline Completed
echo ========================================================
pause
