@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  SONIC — AppXcess / Stella v1
::  Full Windows GPU Server Setup + Run Script
::  Author  : AppXcess Technologies
::  Target  : Windows 10/11 with NVIDIA GPU (CUDA 12.1+)
::  Python  : 3.10.x  (REQUIRED — 3.11+ breaks dependencies)
:: ============================================================

title Sonic - AI Lip-Sync Avatar - AppXcess Technologies

:: ──────────────────────────────────────────────────────────
::  CONFIGURATION
::  HF token is stored in hf_token.txt (gitignored).
::  If missing, this script will prompt you to enter it once
::  and save it automatically for future runs.
:: ──────────────────────────────────────────────────────────
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "CHECKPOINTS_DIR=%SCRIPT_DIR%checkpoints"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"
set "LOG_FILE=%SCRIPT_DIR%setup_log.txt"
set "TOKEN_FILE=%SCRIPT_DIR%hf_token.txt"

:: ── Load or create HuggingFace token ──────────────────────
if exist "%TOKEN_FILE%" (
    set /p HF_TOKEN=<"%TOKEN_FILE%"
    set "HF_TOKEN=%HF_TOKEN: =%"
    if "!HF_TOKEN!"=="" goto :ask_token
    echo  [OK] HuggingFace token loaded from hf_token.txt.
    goto :token_ready
)

:ask_token
echo.
echo  ┌─────────────────────────────────────────────────────┐
echo  │         HUGGINGFACE TOKEN REQUIRED                  │
echo  │                                                     │
echo  │  A HuggingFace access token is needed to download  │
echo  │  the Sonic model weights (~11 GB).                  │
echo  │                                                     │
echo  │  Get your free token at:                            │
echo  │    https://huggingface.co/settings/tokens           │
echo  │                                                     │
echo  │  Before entering your token, make sure you have    │
echo  │  accepted the model licenses (browser, one-time):  │
echo  │    https://huggingface.co/LeonJoe13/Sonic           │
echo  │    https://huggingface.co/stabilityai/              │
echo  │      stable-video-diffusion-img2vid-xt              │
echo  └─────────────────────────────────────────────────────┘
echo.
set "HF_TOKEN="
set /p HF_TOKEN="  Paste your HuggingFace token here and press Enter: "

:: Strip any accidental spaces
set "HF_TOKEN=%HF_TOKEN: =%"

if "%HF_TOKEN%"=="" (
    echo.
    echo  [ERROR] No token entered. Cannot download model weights without a token.
    echo  [ERROR] Re-run this script and paste your token when prompted.
    echo.
    pause
    exit /b 1
)

:: Basic sanity check — HF tokens start with "hf_"
echo %HF_TOKEN% | findstr /r "^hf_" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [WARN] Token does not start with "hf_" — double-check you copied it correctly.
    echo  [WARN] Continuing anyway, but the download may fail if the token is invalid.
    echo.
)

:: Save token to file for future runs
echo %HF_TOKEN%>"%TOKEN_FILE%"
echo  [OK] Token saved to hf_token.txt ^(gitignored — will not be committed^).
echo  [OK] Future runs will load it automatically — no need to enter it again.
echo.

:token_ready


:: ──────────────────────────────────────────────────────────
::  COLORS / HEADER
:: ──────────────────────────────────────────────────────────
echo.
echo  ██████████████████████████████████████████████████████
echo  ██                                                  ██
echo  ██   SONIC  -  AI Lip-Sync Avatar Platform          ██
echo  ██   AppXcess Technologies  ^|  Stella v1            ██
echo  ██   CVPR 2025  ^|  SVD + Whisper + AudioProj        ██
echo  ██                                                  ██
echo  ██████████████████████████████████████████████████████
echo.
echo  [%date% %time%] Starting setup...
echo  Script directory: %SCRIPT_DIR%
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 0 — Write log header
:: ──────────────────────────────────────────────────────────
echo [%date% %time%] ==== SONIC SETUP LOG ==== > "%LOG_FILE%"

:: ──────────────────────────────────────────────────────────
::  STEP 1 — Verify NVIDIA GPU + CUDA
:: ──────────────────────────────────────────────────────────
echo [STEP 1/9] Checking NVIDIA GPU and CUDA...
echo [%date% %time%] Checking GPU... >> "%LOG_FILE%"

nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] nvidia-smi not found.
    echo  [ERROR] An NVIDIA GPU with drivers installed is required.
    echo  [ERROR] Please install the latest NVIDIA Game Ready or Studio drivers.
    echo  [ERROR] Download: https://www.nvidia.com/Download/index.aspx
    echo.
    pause
    exit /b 1
)

echo  [OK] NVIDIA GPU detected.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>>"%LOG_FILE%"
echo  GPU Info:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 2 — Check Python 3.10
:: ──────────────────────────────────────────────────────────
echo [STEP 2/9] Checking Python 3.10...
echo [%date% %time%] Checking Python... >> "%LOG_FILE%"

:: Try to find Python 3.10 in common locations
set "PY310="

:: Check if py launcher is available (most Windows Python installs)
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY310=py -3.10"
    echo  [OK] Found Python 3.10 via py launcher.
    goto :python_found
)

:: Try python3.10 directly
python3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY310=python3.10"
    echo  [OK] Found Python 3.10 as python3.10.
    goto :python_found
)

:: Try python and check version
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo !PY_VER! | findstr /r "^3\.10\." >nul 2>&1
if %errorlevel% equ 0 (
    set "PY310=python"
    echo  [OK] System python is 3.10: !PY_VER!
    goto :python_found
)

:: Python 3.10 not found — attempt to install via winget
echo  [WARN] Python 3.10 not found. Attempting to install via winget...
winget install --id Python.Python.3.10 --source winget --accept-source-agreements --accept-package-agreements --silent
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Could not install Python 3.10 automatically.
    echo  [ERROR] Please install Python 3.10 manually from:
    echo  [ERROR]   https://www.python.org/downloads/release/python-31011/
    echo  [ERROR] During install, check "Add Python to PATH".
    echo  [ERROR] Then re-run this script.
    echo.
    pause
    exit /b 1
)
:: Refresh environment after winget install
set "PATH=%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;%PATH%"
set "PY310=python"

:python_found
echo  Using Python: %PY310%
%PY310% --version
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 3 — Check and install ffmpeg
:: ──────────────────────────────────────────────────────────
echo [STEP 3/9] Checking ffmpeg...
echo [%date% %time%] Checking ffmpeg... >> "%LOG_FILE%"

ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo  [OK] ffmpeg is already installed and on PATH.
    goto :ffmpeg_ok
)

echo  [INFO] ffmpeg not found. Attempting to install via winget...
winget install --id Gyan.FFmpeg --source winget --accept-source-agreements --accept-package-agreements --silent
if %errorlevel% equ 0 (
    :: Refresh PATH to pick up the winget-installed ffmpeg
    set "PATH=%PATH%;%LOCALAPPDATA%\Microsoft\WinGet\Links"
    ffmpeg -version >nul 2>&1
    if %errorlevel% equ 0 (
        echo  [OK] ffmpeg installed successfully via winget.
        goto :ffmpeg_ok
    )
)

:: Manual fallback — download a static build
echo  [INFO] winget install failed or PATH not refreshed. Downloading ffmpeg static build...
set "FFMPEG_DIR=%SCRIPT_DIR%ffmpeg_bin"
if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"

:: Use PowerShell to download ffmpeg
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%FFMPEG_DIR%\ffmpeg.zip' -UseBasicParsing; Write-Host 'Downloaded ffmpeg.' } catch { Write-Host 'Download failed: ' + $_.Exception.Message; exit 1 }"
if %errorlevel% neq 0 (
    echo  [ERROR] Could not download ffmpeg. Please install manually:
    echo  [ERROR]   https://ffmpeg.org/download.html
    echo  [ERROR] Place ffmpeg.exe in a folder that is in your PATH, then re-run.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%FFMPEG_DIR%\ffmpeg.zip' -DestinationPath '%FFMPEG_DIR%' -Force; $dir=(Get-ChildItem '%FFMPEG_DIR%' -Filter 'ffmpeg-*' -Directory | Select-Object -First 1).FullName; Copy-Item \"$dir\bin\ffmpeg.exe\" '%FFMPEG_DIR%\ffmpeg.exe' -Force; Copy-Item \"$dir\bin\ffprobe.exe\" '%FFMPEG_DIR%\ffprobe.exe' -Force"

set "PATH=%FFMPEG_DIR%;%PATH%"
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] ffmpeg still not available after extraction. Please install manually.
    pause
    exit /b 1
)
echo  [OK] ffmpeg installed successfully from static build.

:ffmpeg_ok
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 4 — Create Python virtual environment
:: ──────────────────────────────────────────────────────────
echo [STEP 4/9] Setting up Python virtual environment...
echo [%date% %time%] Creating venv... >> "%LOG_FILE%"

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo  [INFO] Virtual environment already exists at: %VENV_DIR%
    echo  [INFO] Skipping creation, will update packages if needed.
) else (
    echo  [INFO] Creating new venv at: %VENV_DIR%
    %PY310% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        echo  [ERROR] Make sure Python 3.10 has the venv module (usually included by default).
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)
echo.

:: Activate the venv for the rest of this script
call "%VENV_DIR%\Scripts\activate.bat"
echo  [OK] Virtual environment activated.
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 5 — Upgrade pip + install build tools
:: ──────────────────────────────────────────────────────────
echo [STEP 5/9] Upgrading pip and installing build tools...
echo [%date% %time%] Upgrading pip... >> "%LOG_FILE%"

"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)
echo  [OK] pip, setuptools, wheel upgraded.
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 6 — Install PyTorch (CUDA 12.1 build) FIRST
::  CRITICAL: Must install torch before requirements.txt
::  to prevent pip from pulling the CPU-only wheel.
::  torch 2.2.1 is what the repo's requirements.txt specifies.
:: ──────────────────────────────────────────────────────────
echo [STEP 6/9] Installing PyTorch 2.2.1 + CUDA 12.1...
echo [%date% %time%] Installing PyTorch... >> "%LOG_FILE%"
echo  [INFO] This may take 5-10 minutes (downloading ~2 GB)...

:: Check if torch is already installed with CUDA
"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'CUDA', torch.version.cuda)" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('"%PYTHON_EXE%" -c "import torch; print(torch.__version__)"') do set "TORCH_VER=%%i"
    echo  [INFO] PyTorch !TORCH_VER! with CUDA already installed — skipping download.
    goto :torch_done
)

"%PYTHON_EXE%" -m pip install --no-cache-dir ^
    torch==2.2.1+cu121 ^
    torchvision==0.17.1+cu121 ^
    torchaudio==2.2.1+cu121 ^
    --index-url https://download.pytorch.org/whl/cu121 >> "%LOG_FILE%" 2>&1

if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install PyTorch with CUDA 12.1.
    echo  [ERROR] Check your internet connection and try again.
    echo  [ERROR] If problem persists, see: https://pytorch.org/get-started/locally/
    type "%LOG_FILE%"
    pause
    exit /b 1
)
echo  [OK] PyTorch 2.2.1+cu121 installed.

:torch_done
:: Verify CUDA is accessible
"%PYTHON_EXE%" -c "import torch; print('[VERIFY] torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 7 — Install all other Python dependencies
:: ──────────────────────────────────────────────────────────
echo [STEP 7/9] Installing Python dependencies from requirements.txt...
echo [%date% %time%] Installing dependencies... >> "%LOG_FILE%"
echo  [INFO] This may take 5-15 minutes...

:: Install requirements but skip torch/torchaudio/torchvision
:: (already installed above with correct CUDA build)
"%PYTHON_EXE%" -m pip install --no-cache-dir ^
    diffusers==0.29.0 ^
    transformers==4.43.2 ^
    imageio==2.31.1 ^
    "imageio-ffmpeg==0.5.1" ^
    "gradio==3.50.0" ^
    "omegaconf==2.3.0" ^
    "tqdm==4.65.2" ^
    "librosa==0.10.2.post1" ^
    "einops==0.7.0" >> "%LOG_FILE%" 2>&1

if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install core dependencies. Check %LOG_FILE% for details.
    pause
    exit /b 1
)

:: Additional dependencies used by the codebase but missing from requirements.txt
"%PYTHON_EXE%" -m pip install --no-cache-dir ^
    opencv-python ^
    Pillow ^
    pydub ^
    scipy ^
    numpy ^
    accelerate ^
    huggingface_hub ^
    safetensors ^
    ftfy ^
    regex ^
    requests >> "%LOG_FILE%" 2>&1

if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install supplemental dependencies. Check %LOG_FILE% for details.
    pause
    exit /b 1
)

:: Install huggingface CLI for weight download
"%PYTHON_EXE%" -m pip install --no-cache-dir "huggingface_hub[cli]" >> "%LOG_FILE%" 2>&1

echo  [OK] All Python dependencies installed.
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 8 — Download model weights from HuggingFace
::  Weight structure (under checkpoints/):
::    checkpoints/
::    ├── Sonic/
::    │   ├── unet.pth          (~500 MB)  — from LeonJoe13/Sonic
::    │   ├── audio2token.pth   (~100 MB)  — from LeonJoe13/Sonic
::    │   └── audio2bucket.pth  (~100 MB)  — from LeonJoe13/Sonic
::    ├── RIFE/
::    │   └── flownet.pkl       (~50 MB)   — from LeonJoe13/Sonic
::    ├── yoloface_v5m.pt       (~50 MB)   — from LeonJoe13/Sonic
::    ├── stable-video-diffusion-img2vid-xt/  (~8 GB) — from stabilityai
::    └── whisper-tiny/          (~150 MB) — from openai
:: ──────────────────────────────────────────────────────────
echo [STEP 8/9] Downloading model weights from HuggingFace...
echo [%date% %time%] Downloading weights... >> "%LOG_FILE%"

:: Create checkpoints directory
if not exist "%CHECKPOINTS_DIR%" mkdir "%CHECKPOINTS_DIR%"

:: ── 8a: LeonJoe13/Sonic weights (unet, audio adapters, RIFE, yoloface) ──
echo  [INFO] Downloading Sonic adapter weights (LeonJoe13/Sonic)...
set "SONIC_WEIGHTS_DONE=%CHECKPOINTS_DIR%\Sonic\unet.pth"
if exist "%SONIC_WEIGHTS_DONE%" (
    echo  [SKIP] Sonic weights already downloaded.
) else (
    "%VENV_DIR%\Scripts\huggingface-cli" download LeonJoe13/Sonic ^
        --local-dir "%CHECKPOINTS_DIR%" ^
        --token "%HF_TOKEN%" ^
        --quiet >> "%LOG_FILE%" 2>&1
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to download LeonJoe13/Sonic weights.
        echo  [ERROR] Ensure you have accepted the model license at:
        echo  [ERROR]   https://huggingface.co/LeonJoe13/Sonic
        echo  [ERROR] and that the HF_TOKEN has read access.
        echo  [ERROR] Check %LOG_FILE% for full error details.
        pause
        exit /b 1
    )
    echo  [OK] Sonic adapter weights downloaded.
)
echo.

:: ── 8b: Stable Video Diffusion SVD XT base model ──
echo  [INFO] Downloading SVD base model (stabilityai/stable-video-diffusion-img2vid-xt)...
echo  [INFO] This is ~8 GB and may take 20-60 minutes depending on connection speed.
set "SVD_DONE=%CHECKPOINTS_DIR%\stable-video-diffusion-img2vid-xt\model_index.json"
if exist "%SVD_DONE%" (
    echo  [SKIP] SVD weights already downloaded.
) else (
    "%VENV_DIR%\Scripts\huggingface-cli" download stabilityai/stable-video-diffusion-img2vid-xt ^
        --local-dir "%CHECKPOINTS_DIR%\stable-video-diffusion-img2vid-xt" ^
        --token "%HF_TOKEN%" ^
        --quiet >> "%LOG_FILE%" 2>&1
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to download SVD XT weights.
        echo  [ERROR] Ensure you have accepted the model license at:
        echo  [ERROR]   https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
        echo  [ERROR] Check %LOG_FILE% for full error details.
        pause
        exit /b 1
    )
    echo  [OK] SVD XT base model downloaded.
)
echo.

:: ── 8c: Whisper Tiny audio encoder ──
echo  [INFO] Downloading Whisper Tiny (openai/whisper-tiny)...
set "WHISPER_DONE=%CHECKPOINTS_DIR%\whisper-tiny\config.json"
if exist "%WHISPER_DONE%" (
    echo  [SKIP] Whisper Tiny already downloaded.
) else (
    "%VENV_DIR%\Scripts\huggingface-cli" download openai/whisper-tiny ^
        --local-dir "%CHECKPOINTS_DIR%\whisper-tiny" ^
        --token "%HF_TOKEN%" ^
        --quiet >> "%LOG_FILE%" 2>&1
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to download Whisper Tiny weights.
        echo  [ERROR] Check %LOG_FILE% for full error details.
        pause
        exit /b 1
    )
    echo  [OK] Whisper Tiny downloaded.
)
echo.

:: ── 8d: Verify checkpoint structure ──
echo  [INFO] Verifying checkpoint structure...
set "VERIFY_OK=1"

if not exist "%CHECKPOINTS_DIR%\Sonic\unet.pth"          ( echo  [MISSING] checkpoints\Sonic\unet.pth           & set "VERIFY_OK=0" )
if not exist "%CHECKPOINTS_DIR%\Sonic\audio2token.pth"   ( echo  [MISSING] checkpoints\Sonic\audio2token.pth    & set "VERIFY_OK=0" )
if not exist "%CHECKPOINTS_DIR%\Sonic\audio2bucket.pth"  ( echo  [MISSING] checkpoints\Sonic\audio2bucket.pth   & set "VERIFY_OK=0" )
if not exist "%CHECKPOINTS_DIR%\RIFE\flownet.pkl"         ( echo  [MISSING] checkpoints\RIFE\flownet.pkl          & set "VERIFY_OK=0" )
if not exist "%CHECKPOINTS_DIR%\yoloface_v5m.pt"         ( echo  [MISSING] checkpoints\yoloface_v5m.pt           & set "VERIFY_OK=0" )
if not exist "%CHECKPOINTS_DIR%\stable-video-diffusion-img2vid-xt\model_index.json" ( echo  [MISSING] checkpoints\stable-video-diffusion-img2vid-xt\model_index.json & set "VERIFY_OK=0" )
if not exist "%CHECKPOINTS_DIR%\whisper-tiny\config.json" ( echo  [MISSING] checkpoints\whisper-tiny\config.json  & set "VERIFY_OK=0" )

if "%VERIFY_OK%"=="0" (
    echo.
    echo  [ERROR] Some model checkpoints are missing (see above).
    echo  [ERROR] The weight download may have been incomplete.
    echo  [ERROR] Please re-run this script or download manually:
    echo  [ERROR]   huggingface-cli download LeonJoe13/Sonic --local-dir checkpoints
    echo  [ERROR]   huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints\stable-video-diffusion-img2vid-xt
    echo  [ERROR]   huggingface-cli download openai/whisper-tiny --local-dir checkpoints\whisper-tiny
    pause
    exit /b 1
)
echo  [OK] All checkpoints verified.
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 8e — Apply Critical Code Patches
::  Fixes two known bugs in the original Sonic repo:
::  1. Double os.path.join bug on yoloface det_path (sonic.py line 200)
::  2. enable_attention_slicing(1) causing slow Python loop (if present)
:: ──────────────────────────────────────────────────────────
echo  [INFO] Applying critical bug fixes to sonic.py...

"%PYTHON_EXE%" -c "
import re, os, shutil

sonic_path = os.path.join(r'%SCRIPT_DIR%', 'sonic.py')
backup_path = sonic_path + '.original_backup'

with open(sonic_path, 'r', encoding='utf-8') as f:
    content = f.read()

changed = False

# FIX 1: Double os.path.join bug on yoloface det_path
# Original buggy line:
#   det_path = os.path.join(BASE_DIR, os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt'))
# Correct:
#   det_path = os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt')
old = 'det_path = os.path.join(BASE_DIR, os.path.join(BASE_DIR, '
if old in content:
    content = content.replace(
        \"det_path = os.path.join(BASE_DIR, os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt'))\",
        \"det_path = os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt')\"
    )
    changed = True
    print('[PATCH 1] Fixed double os.path.join bug on yoloface det_path.')
else:
    print('[PATCH 1] Double os.path.join bug not present (already fixed or different version).')

# FIX 2: Remove enable_attention_slicing(1) if present (replaces fast attention with slow Python loop)
if 'enable_attention_slicing' in content:
    content = re.sub(r'.*enable_attention_slicing.*\n', '', content)
    changed = True
    print('[PATCH 2] Removed enable_attention_slicing call.')
else:
    print('[PATCH 2] enable_attention_slicing not present — no action needed.')

if changed:
    if not os.path.exists(backup_path):
        shutil.copy2(sonic_path, backup_path)
        print('[INFO] Original sonic.py backed up to sonic.py.original_backup')
    with open(sonic_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('[OK] Patches applied successfully.')
else:
    print('[OK] No patches were necessary.')
"

if %errorlevel% neq 0 (
    echo  [WARN] Could not apply patches automatically. The script may still run but check sonic.py manually.
)
echo.

:: ──────────────────────────────────────────────────────────
::  STEP 9 — Create output directories and run final check
:: ──────────────────────────────────────────────────────────
echo [STEP 9/9] Final environment verification and launch...
echo [%date% %time%] Final check... >> "%LOG_FILE%"

:: Create output directories used by gradio_app.py
if not exist "%SCRIPT_DIR%tmp_path" mkdir "%SCRIPT_DIR%tmp_path"
if not exist "%SCRIPT_DIR%res_path" mkdir "%SCRIPT_DIR%res_path"

:: Final dependency check
echo  [INFO] Running final import check...
"%PYTHON_EXE%" -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB')
import diffusers; print(f'diffusers: {diffusers.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
import gradio; print(f'gradio: {gradio.__version__}')
import cv2; print(f'opencv: {cv2.__version__}')
import librosa; print(f'librosa: {librosa.__version__}')
import imageio; print(f'imageio: {imageio.__version__}')
print('All imports OK.')
"
if %errorlevel% neq 0 (
    echo  [ERROR] Import check failed. Check %LOG_FILE% for details.
    pause
    exit /b 1
)
echo.

:: ──────────────────────────────────────────────────────────
::  LAUNCH — Start Gradio App
:: ──────────────────────────────────────────────────────────
echo  ══════════════════════════════════════════════════════
echo  [LAUNCH] Starting Sonic Gradio App...
echo  ══════════════════════════════════════════════════════
echo.
echo  Setup complete! Sonic is now loading the models.
echo  This initial model load takes 2-5 minutes.
echo.
echo  Once loaded, open your browser at:
echo    http://localhost:8081
echo    (or the public share URL shown below if share=True)
echo.
echo  Upload a portrait image and an audio file to generate
echo  a lip-synced talking-head video.
echo.
echo  Press Ctrl+C to stop the server.
echo.
echo  ══════════════════════════════════════════════════════
echo  [%date% %time%] Launching gradio_app.py >> "%LOG_FILE%"

set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%SCRIPT_DIR%.hf_cache"

cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" gradio_app.py

:: If it exits (e.g. user hits Ctrl+C or an error), pause so the user can see the output
echo.
echo  [INFO] Sonic has stopped. If this was unexpected, check the output above.
echo  [INFO] Setup log is at: %LOG_FILE%
pause
