@echo off
setlocal
cd /d "%~dp0"
echo Starting the full ChudGPT-Public CUDA training pipeline.
echo Progress is written to the reports folder.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_pipeline.ps1" -Device cuda
if errorlevel 1 (
  echo Training failed. Read the newest reports\public_training_*.log file.
  pause
  exit /b 1
)
echo Training and evaluation completed successfully. The API checkpoint is ready.
pause
