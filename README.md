# Sonic — AI Lip-Sync Avatar Platform
### AppXcess Technologies · Stella v1 · CVPR 2025

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://jixiaozhong.github.io/Sonic/)
[![Paper](https://img.shields.io/badge/Paper-CVPR_2025-red)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ji_Sonic_Shifting_Focus_to_Global_Audio_Perception_in_Portrait_Animation_CVPR_2025_paper.pdf)
[![HuggingFace Demo](https://img.shields.io/badge/Space-ZeroGPU-orange?logo=Gradio)](https://huggingface.co/spaces/xiaozhongji/Sonic)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgreen)](https://raw.githubusercontent.com/jixiaozhong/Sonic/refs/heads/main/LICENSE)

**Take a portrait photo + audio → generate a photorealistic lip-synced talking-head video.**

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [How It Works — Model Deep Dive](#3-how-it-works--model-deep-dive)
4. [System Requirements](#4-system-requirements)
5. [File Structure](#5-file-structure)
6. [Quick Start — Windows GPU Server](#6-quick-start--windows-gpu-server)
7. [Manual Installation](#7-manual-installation)
8. [Weight Downloads](#8-weight-downloads)
9. [Running the App](#9-running-the-app)
10. [CLI Usage (demo.py)](#10-cli-usage-demopy)
11. [Configuration Reference](#11-configuration-reference)
12. [Known Issues & Fixes](#12-known-issues--fixes)
13. [Performance Benchmarks](#13-performance-benchmarks)
14. [API / Integration Notes](#14-api--integration-notes)
15. [License](#15-license)
16. [Citation](#16-citation)

---

## 1. Project Overview

**Sonic** (CVPR 2025) is a portrait animation model that generates realistic, temporally coherent talking-head videos from a single still image and an audio track. It was selected as the core model for **AppXcess Technologies' Stella v1** avatar platform after extensive evaluation against LatentSync, EchoMimic, Ditto, and SadTalker.

### Why Sonic?

| Model | Status | Reason |
|---|---|---|
| LatentSync (ByteDance 2024) | ❌ Abandoned | Cannot animate still photos — requires video input |
| EchoMimic V1 (Ant Group, AAAI 2025) | ❌ Not deployed | 14 GB VRAM minimum; Colab-only viable |
| Ditto TalkingHead (Ant Group, CVPR 2025) | ❌ Abandoned | MotionStitch locks expressions every 70 frames |
| SadTalker (CVPR 2023) | 🔶 Local fallback | Good quality; MIT license; too slow for production |
| **Sonic (CVPR 2025)** | ✅ **ACTIVE** | Best quality; SVD-based; stable temporal coherence |

### What Stella v1 Delivers

- Upload a **portrait image** (JPG/PNG) of any person
- Upload an **audio file** (WAV/MP3) with speech
- Receive a **lip-synced MP4 video** with natural head motion and eye blinks
- Served via a **Gradio web UI** on port 8081

---

## 2. Architecture

```
User Browser
     │
     └── Gradio UI  (gradio_app.py — port 8081)
              │
              └── Sonic Class  (sonic.py)
                       │
                       ├── Face Detection: YOLOFace v5m
                       │     └── Locates and crops the face region
                       │
                       ├── Audio Encoding: OpenAI Whisper Tiny
                       │     └── Converts audio waveform → hidden state embeddings
                       │
                       ├── Audio Adapters (Sonic-specific, fine-tuned)
                       │     ├── AudioProjModel  (audio2token.pth)
                       │     │     └── Maps Whisper embeddings → audio tokens for UNet
                       │     └── Audio2bucketModel  (audio2bucket.pth)
                       │           └── Predicts motion bucket (head movement intensity)
                       │
                       ├── Image Encoding: CLIP Vision (from SVD)
                       │     └── Extracts appearance features from portrait
                       │
                       ├── UNet: SVD SpatioTemporal + IP-Adapter (unet.pth)
                       │     └── Denoises video latents conditioned on audio + image
                       │
                       ├── VAE: AutoencoderKLTemporalDecoder (from SVD)
                       │     └── Decodes latents → pixel frames
                       │
                       └── RIFE Frame Interpolation (checkpoints/RIFE/)
                             └── 2× temporal upsampling for smoother motion
```

**Pipeline summary:**
1. Face detected and cropped from portrait
2. Audio chunked into 750-frame windows, encoded by Whisper
3. Per-chunk: motion bucket predicted → audio tokens computed
4. SVD UNet denoises all chunks in batch, conditioned on audio + portrait
5. RIFE doubles frame rate (12.5 fps → 25 fps output)
6. FFmpeg muxes video frames + original audio → final `.mp4`

---

## 3. How It Works — Model Deep Dive

### Stable Video Diffusion (SVD) Base
Sonic builds on `stable-video-diffusion-img2vid-xt` — Stability AI's image-to-video diffusion model. SVD uses a 3D UNet with temporal attention layers to generate temporally coherent video from a single reference frame.

### Audio Conditioning
The key innovation of Sonic is **global audio perception**: instead of conditioning only on local phoneme windows, Sonic processes the entire audio context through:
- **Whisper Tiny:** Extracts 384-channel hidden states at every transformer layer
- **AudioProjModel:** A cross-attention projection mapping Whisper's 32 hidden layers into 32 audio tokens per clip segment (sequence length 10)
- **Audio2bucketModel:** A regression head predicting motion intensity (mapped to SVD's motion bucket parameter), controlling how much the head moves

### IP-Adapter Integration
Sonic injects audio conditioning into the SVD UNet via IP-Adapter-style cross-attention layers at scale 32. This is added on top of the pre-trained SVD weights — allowing fine-tuned audio control without destroying appearance conditioning.

### RIFE Frame Interpolation
After generation at 12.5 fps, the optional RIFE (Real-time Intermediate Flow Estimation) model inserts synthesized intermediate frames between every consecutive pair, producing 25 fps output with smoother motion.

---

## 4. System Requirements

### Minimum (will work, slower)
| Component | Requirement |
|---|---|
| OS | Windows 10 / Windows 11 (64-bit) |
| GPU | NVIDIA GPU with **12 GB VRAM** |
| CUDA Driver | 525+ (supports CUDA 12.1) |
| RAM | 16 GB |
| Disk | 50 GB free (11 GB weights + model + venv) |
| Python | **3.10.x** (NOT 3.11, 3.12, 3.13, 3.14) |
| Internet | Required for first-run weight download (~11 GB) |

### Recommended (tested)
| Component | Specification |
|---|---|
| GPU | NVIDIA T4 (16 GB VRAM) or RTX 4060+ (8 GB min) |
| RAM | 32 GB |
| Disk | 100 GB SSD |
| Connection | 100 Mbps+ for initial weight download |

### VRAM vs. Performance
| VRAM | Resolution | Inference Time (25 steps, 20s audio) |
|---|---|---|
| 8 GB (RTX 4060) | 256×320 | ~3.6 min |
| 8 GB (RTX 4060) | 512×512 | OOM — reduce decode_chunk_size |
| 16 GB (T4) | 512×512 | ~8–15 min |
| 24 GB (RTX 4090) | 512×512 | ~4–6 min |

> **Note:** The bottleneck is 3D temporal convolutions (scale as H×W), NOT attention. At 512px on 8 GB VRAM you will hit OOM — use `decode_chunk_size=4` or keep resolution at 256px.

---

## 5. File Structure

```
Sonic/
├── run_project.bat              ← ONE-CLICK setup + launch (Windows GPU server)
├── demo.py                      ← CLI inference script
├── gradio_app.py                ← Gradio web UI (port 8081)
├── sonic.py                     ← Core Sonic class (model loader + inference)
├── requirements.txt             ← Python dependencies
├── demo.sh                      ← Linux demo launcher (not used on Windows)
│
├── config/
│   └── inference/
│       └── sonic.yaml           ← Runtime hyperparameters (steps, fps, etc.)
│
├── src/
│   ├── dataset/
│   │   ├── test_preprocess.py   ← Image/audio preprocessing (face crop, audio features)
│   │   └── face_align/
│   │       ├── align.py         ← AlignImage wrapper around YOLOFace
│   │       └── yoloface.py      ← YOLOFace v5m face detector
│   │
│   ├── models/
│   │   ├── audio_adapter/
│   │   │   ├── audio_proj.py    ← AudioProjModel (Whisper → UNet tokens)
│   │   │   └── audio_to_bucket.py ← Audio2bucketModel (motion bucket predictor)
│   │   └── base/
│   │       └── unet_spatio_temporal_condition.py ← SVD UNet + IP-Adapter injection
│   │
│   ├── pipelines/
│   │   └── pipeline_sonic.py    ← Main diffusion pipeline (batched chunked inference)
│   │
│   └── utils/
│       ├── util.py              ← save_videos_grid, seed_everything
│       ├── mask_processer.py    ← Face mask generation
│       └── RIFE/
│           ├── RIFE_HDv3.py     ← RIFE model wrapper
│           ├── IFNet_HDv3.py    ← RIFE flow estimation network
│           └── warplayer.py     ← Frame warping utility
│
├── checkpoints/                 ← Model weights (downloaded at first run, ~11 GB)
│   ├── Sonic/
│   │   ├── unet.pth             ← Sonic UNet adapter (~500 MB)
│   │   ├── audio2token.pth      ← Audio projection model (~100 MB)
│   │   └── audio2bucket.pth     ← Motion bucket predictor (~100 MB)
│   ├── RIFE/
│   │   └── flownet.pkl          ← RIFE interpolation network (~50 MB)
│   ├── yoloface_v5m.pt          ← YOLOFace detector (~50 MB)
│   ├── stable-video-diffusion-img2vid-xt/  ← SVD XT base model (~8 GB)
│   └── whisper-tiny/            ← Whisper Tiny audio encoder (~150 MB)
│
├── examples/
│   ├── image/                   ← Sample portrait images
│   └── wav/                     ← Sample audio files
│
├── tmp_path/                    ← Gradio temp files (auto-created)
├── res_path/                    ← Gradio output videos (auto-created)
├── setup_log.txt                ← Setup and install log (auto-created by bat)
└── .venv/                       ← Python 3.10 virtual environment (auto-created)
```

---

## 6. Quick Start — Windows GPU Server

> This is the **recommended path** for deploying on a Windows GPU server.

### Prerequisites (one-time)
1. Install [Python 3.10](https://www.python.org/downloads/release/python-31011/) — check "Add Python to PATH"
2. Install [NVIDIA GPU drivers](https://www.nvidia.com/Download/index.aspx) (version 525+)
3. Accept the HuggingFace model licenses (browser login required):
   - [LeonJoe13/Sonic](https://huggingface.co/LeonJoe13/Sonic)
   - [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)

### Run

```powershell
# 1. Clone the repo (if not already done)
git clone https://github.com/vishnupriyanpr/Sonic.git
cd Sonic

# 2. Right-click run_project.bat → "Run as Administrator"
#    OR from an elevated PowerShell:
.\run_project.bat
```

**The script does everything automatically:**
1. ✅ Checks for NVIDIA GPU + CUDA
2. ✅ Checks for / installs Python 3.10 (via winget)
3. ✅ Checks for / installs ffmpeg (via winget or static download)
4. ✅ Creates a Python 3.10 virtual environment at `.venv/`
5. ✅ Installs PyTorch 2.2.1 + CUDA 12.1 (correct GPU build)
6. ✅ Installs all Python dependencies
7. ✅ Downloads all model weights (~11 GB) from HuggingFace
8. ✅ Applies critical bug patches to `sonic.py`
9. ✅ Launches the Gradio UI at `http://localhost:8081`

> **Re-runs are fast.** All steps that already completed (venv, weights, patches) are skipped automatically. Only model loading runs every time (~2–5 min).

---

## 7. Manual Installation

If you prefer step-by-step control:

```powershell
# 1. Create and activate venv with Python 3.10
py -3.10 -m venv .venv
.venv\Scripts\activate

# 2. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA 12.1 FIRST (critical — prevents CPU-only install)
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining requirements
pip install diffusers==0.29.0 transformers==4.43.2 imageio==2.31.1 imageio-ffmpeg==0.5.1 `
    gradio==3.50.0 omegaconf==2.3.0 tqdm==4.65.2 "librosa==0.10.2.post1" einops==0.7.0

# 5. Install supplemental deps (not in requirements.txt but required by codebase)
pip install opencv-python Pillow pydub scipy numpy accelerate `
    huggingface_hub safetensors ftfy regex requests "huggingface_hub[cli]"
```

---

## 8. Weight Downloads

Total disk usage: **~11.4 GB**

```powershell
# Set your HuggingFace token
$env:HF_TOKEN = "your_hf_token_here"

# Download Sonic adapter weights (unet, audio adapters, RIFE, yoloface)
huggingface-cli download LeonJoe13/Sonic `
    --local-dir checkpoints `
    --token $env:HF_TOKEN

# Download SVD XT base model (~8 GB — takes 20-60 min)
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt `
    --local-dir checkpoints/stable-video-diffusion-img2vid-xt `
    --token $env:HF_TOKEN

# Download Whisper Tiny audio encoder (~150 MB)
huggingface-cli download openai/whisper-tiny `
    --local-dir checkpoints/whisper-tiny `
    --token $env:HF_TOKEN
```

### Expected checkpoint structure after download
```
checkpoints/
├── Sonic/
│   ├── unet.pth             ~500 MB
│   ├── audio2token.pth      ~100 MB
│   └── audio2bucket.pth     ~100 MB
├── RIFE/
│   └── flownet.pkl          ~50 MB
├── yoloface_v5m.pt          ~50 MB
├── stable-video-diffusion-img2vid-xt/   ~8 GB
│   ├── model_index.json
│   ├── vae/
│   ├── image_encoder/
│   ├── unet/
│   └── scheduler/
└── whisper-tiny/            ~150 MB
    ├── config.json
    ├── model.safetensors
    └── ...
```

---

## 9. Running the App

### Gradio Web UI

```powershell
# Activate venv first
.venv\Scripts\activate

# Launch Gradio (model load: ~2–5 min, then ready)
python gradio_app.py
```

Open browser at: **http://localhost:8081**

**UI controls:**
- **Upload Image** — Portrait photo (JPG or PNG). Face must be clearly visible and front-facing.
- **Upload Audio** — Speech audio (WAV or MP3). Longer audio = longer generation time.
- **Dynamic Scale** — Slider (0.5–2.0). Controls head movement intensity. `1.0` = natural, `2.0` = exaggerated.

**Output:** MP4 video at 512×512 px with the portrait speaking the uploaded audio.

---

## 10. CLI Usage (demo.py)

```powershell
# Activate venv
.venv\Scripts\activate

# Basic usage
python demo.py "path\to\portrait.jpg" "path\to\audio.wav" "path\to\output.mp4"

# With options
python demo.py "portrait.jpg" "speech.wav" "output.mp4" `
    --dynamic_scale 1.2 `    # 0.5–2.0, head movement intensity
    --crop `                  # Auto-crop to face region before inference
    --seed 42                 # Fixed seed for reproducibility
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `image_path` | str | required | Path to input portrait image |
| `audio_path` | str | required | Path to input audio file (WAV/MP3) |
| `output_path` | str | required | Path for output MP4 video |
| `--dynamic_scale` | float | 1.0 | Head motion intensity (0.5 = subtle, 2.0 = strong) |
| `--crop` | flag | False | Auto-crop face region before inference |
| `--seed` | int | None | Random seed for reproducibility |

---

## 11. Configuration Reference

**File:** `config/inference/sonic.yaml`

```yaml
# Model paths (relative to Sonic repo root)
pretrained_model_name_or_path: "checkpoints/stable-video-diffusion-img2vid-xt"
unet_checkpoint_path: "checkpoints/Sonic/unet.pth"
audio2token_checkpoint_path: "checkpoints/Sonic/audio2token.pth"
audio2bucket_checkpoint_path: "checkpoints/Sonic/audio2bucket.pth"

# Precision
weight_dtype: 'fp16'          # fp16 (fast, less VRAM), fp32 (slow, more precise)

# Generation
num_inference_steps: 25       # Denoising steps. More = better quality, slower. Min 10.
n_sample_frames: 25           # Frames per chunk (must be ≤ SVD max)
fps: 12.5                     # Base FPS (doubled to 25 by RIFE if use_interframe=True)
decode_chunk_size: 8          # VAE decode batch size. Reduce to 4 if OOM.
motion_bucket_scale: 1.0      # Multiplier on predicted motion bucket

# Audio / temporal
frame_num: 10000              # Max frames (effectively unlimited)
step: 2                       # Audio stride per chunk
overlap: 0                    # Chunk overlap frames

# Guidance
min_appearance_guidance_scale: 2.0   # Appearance CFG min
max_appearance_guidance_scale: 2.0   # Appearance CFG max
audio_guidance_scale: 7.5           # Audio CFG strength

# Misc
i2i_noise_strength: 1.0       # Image-to-image noise (1.0 = full generation)
noise_aug_strength: 0.00      # Augmentation noise
ip_audio_scale: 1.0           # IP-Adapter audio scale
area: 1.1                     # Face mask area expansion ratio
image_size: 512               # Target resolution (short side)

# Frame interpolation
use_interframe: True          # Enable RIFE 2× frame interpolation
seed: 72589                   # Default random seed
```

### Tuning Tips
- **Out of memory (OOM):** Reduce `decode_chunk_size` from 8 → 4. Or set `image_size: 256` in sonic.yaml.
- **Too slow:** Reduce `num_inference_steps` from 25 → 10. Quality will decrease slightly.
- **Jerky motion:** Increase `dynamic_scale` (or `motion_bucket_scale`). Enable `use_interframe: True`.
- **Blurry output:** Increase `num_inference_steps`. Use FP32 (`weight_dtype: fp32`) if VRAM allows.
- **Face not detected:** Image must have a clear, front-facing, well-lit face. Try `--crop` flag.

---

## 12. Known Issues & Fixes

### Bug 1: Double `os.path.join` for yoloface path (Critical)
**Symptom:** `FileNotFoundError` for `yoloface_v5m.pt` even when it exists.

**Root cause:** `sonic.py` line 200 has a double join:
```python
# BUGGY (original)
det_path = os.path.join(BASE_DIR, os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt'))

# FIXED
det_path = os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt')
```
**Fix:** `run_project.bat` applies this patch automatically at first run.

---

### Bug 2: `enable_attention_slicing(1)` (Performance)
**Symptom:** Generation is extremely slow — 10-20× slower than expected.

**Root cause:** This call replaces the optimized `AttnProcessor2_0` with a slow Python loop.

**Fix:** Remove any call to `.enable_attention_slicing()` in the pipeline. `run_project.bat` removes it automatically.

---

### Bug 3: PyTorch installed as CPU-only build
**Symptom:** `torch.cuda.is_available()` returns `False` despite having an NVIDIA GPU.

**Root cause:** Running `pip install torch` without specifying the CUDA index URL installs the CPU-only PyTorch build.

**Fix:**
```powershell
# Uninstall the wrong build
pip uninstall torch torchvision torchaudio -y

# Reinstall with CUDA 12.1
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
```

---

### Bug 4: xformers breaks CUDA PyTorch
**Symptom:** After installing `xformers` via `pip install xformers`, PyTorch reverts to CPU-only.

**Fix:**
```powershell
pip uninstall xformers -y
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
```
> **Rule:** Never `pip install xformers` in this environment. It is not required by Sonic.

---

### Bug 5: System Python 3.14 is incompatible
**Symptom:** Various import errors, `SyntaxError`, or `ModuleNotFoundError` for `diffusers`, `transformers`, etc.

**Root cause:** This project's dependencies do not support Python 3.11+. All packages must be installed into a **Python 3.10** environment.

**Fix:** Always activate `.venv\Scripts\activate` before running anything. The `run_project.bat` creates and activates this automatically.

---

### Bug 6: ffmpeg not on PATH — video has no audio
**Symptom:** Output file is silent, or `ffmpeg` command fails during `pipe.process()`.

**Fix:** Install ffmpeg and ensure it is on PATH:
```powershell
winget install Gyan.FFmpeg
# Restart terminal to refresh PATH
```

---

## 13. Performance Benchmarks

### Local Test Results (2026-06-18, RTX 4060 8GB)

| Config | Resolution | Audio Length | Inference Time | Output Size |
|---|---|---|---|---|
| 5 steps, 256px, no RIFE | 256×320 | 19.96s | 3.6 min | 1.05 MB |
| 25 steps, 512px, RIFE | 512×512 | 10s | ~OOM on 8GB | — |

### Estimated Performance on Common GPUs

| GPU | VRAM | Resolution | Steps | Estimated Time |
|---|---|---|---|---|
| RTX 4060 | 8 GB | 256×320 | 25 | ~12–15 min |
| RTX 4070 Ti | 12 GB | 512×512 | 25 | ~20–30 min |
| T4 (Google Cloud) | 16 GB | 512×512 | 25 | ~8–15 min |
| A100 | 40 GB | 512×512 | 25 | ~3–5 min |
| RTX 4090 | 24 GB | 512×512 | 25 | ~4–6 min |

> The bottleneck is **3D temporal convolutions** (scale with H×W), not attention. Reducing resolution has a large impact on speed.

---

## 14. API / Integration Notes

The Gradio app runs at `http://0.0.0.0:8081` and exposes a `share=True` public URL via Gradio tunneling.

### Programmatic Usage

```python
from sonic import Sonic

# Initialize (loads all models — takes 2–5 min)
pipe = Sonic(device_id=0, enable_interpolate_frame=True)

# Optional: Detect and crop face
face_info = pipe.preprocess("portrait.jpg", expand_ratio=0.5)
if face_info['face_num'] > 0:
    pipe.crop_image("portrait.jpg", "portrait_cropped.png", face_info['crop_bbox'])

# Generate video
result = pipe.process(
    image_path="portrait_cropped.png",
    audio_path="speech.wav",
    output_path="output/result.mp4",
    min_resolution=512,       # Short side resolution
    inference_steps=25,       # Denoising steps
    dynamic_scale=1.0,        # Motion intensity
    seed=42                   # Optional fixed seed
)
# Returns 0 on success, -1 if no face detected
```

### Output File Naming
Gradio generates outputs at: `res_path/{image_md5}_{audio_md5}_{dynamic_scale}.mp4`

Results are cached — uploading the same image + audio + scale returns the cached video instantly.

---

## 15. License

This project uses the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license.

- ✅ Allowed: Research, education, personal projects
- ❌ Not allowed: Commercial products, SaaS, monetized services

> **For commercial use**, Tencent Cloud offers a licensed commercial API:
> [Tencent Cloud Video Creation Large Model](https://cloud.tencent.com/product/vclm)

---

## 16. Citation

```bibtex
@inproceedings{ji2025sonic,
  title={Sonic: Shifting focus to global audio perception in portrait animation},
  author={Ji, Xiaozhong and Hu, Xiaobin and Xu, Zhihong and Zhu, Junwei and Lin, Chuming
          and He, Qingdong and Zhang, Jiangning and Luo, Donghao and Chen, Yi and Lin, Qin and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={193--203},
  year={2025}
}
```

---

<div align="center">

Built with ❤️ by **AppXcess Technologies**

[![Star History Chart](https://api.star-history.com/svg?repos=jixiaozhong/Sonic&type=Date)](https://star-history.com/#jixiaozhong/Sonic&Date)

</div>
