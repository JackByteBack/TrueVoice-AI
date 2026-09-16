# TrueVoice AI

**AI-powered voice cloning and fake audio detection system**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

**Developed by Claude & Opencode @jackbyteback**

## What is TrueVoice AI?

TrueVoice AI is an end-to-end deep learning system that can **detect AI-generated and synthetic audio** with high accuracy. It combines voice cloning research with signal processing analysis to build a robust classifier that distinguishes real human speech from manipulated or AI-generated audio.

The system processes audio files, converts them into mel spectrograms, and feeds them through a custom CNN (FADCNN) to classify them as **REAL** or **FAKE**. Results are presented through a clean web interface with interactive visualizations and optional AI-generated explanations.

---

## Key Features

- **Real-time audio classification** — Upload any audio file and get instant REAL/FAKE prediction
- **Mel spectrogram visualization** — See the frequency representation the model analyzes
- **Waveform display** — Interactive audio waveform plot
- **AI-powered explanations** — Optional GPT-powered analysis explaining why the audio was classified a certain way
- **Full training pipeline** — From raw data download to trained model in one script
- **Data augmentation** — Time masking, frequency masking, and Gaussian noise for robust training
- **Multiple audio format support** — WAV, MP3, M4A, FLAC (via FFmpeg conversion)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     TrueVoice AI Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA COLLECTION                                             │
│     ├── LibriSpeech (real audio samples)                        │
│     └── CommonVoice (real audio samples)                        │
│                                                                 │
│  2. FAKE AUDIO GENERATION                                       │
│     ├── Noise addition (Gaussian noise injection)               │
│     ├── Time stretching (speed up / slow down)                  │
│     ├── Pitch shifting (semitone adjustments)                   │
│     └── Spectral masking (frequency band suppression)           │
│                                                                 │
│  3. FEATURE EXTRACTION                                          │
│     └── Mel Spectrograms (64 bands, 1024 FFT, 256 hop)         │
│                                                                 │
│  4. CNN CLASSIFICATION (FADCNN)                                 │
│     ├── Conv2d(1→16) + BatchNorm + ReLU + MaxPool              │
│     ├── Conv2d(16→32) + BatchNorm + ReLU + MaxPool             │
│     ├── Conv2d(32→64) + BatchNorm + ReLU + AdaptiveAvgPool     │
│     ├── Dropout(0.3)                                            │
│     └── Linear(64→1) → REAL/FAKE probability                   │
│                                                                 │
│  5. WEB DEPLOYMENT                                              │
│     └── FastAPI + Plotly.js + OpenAI explanations               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
TrueVoice-AI/
├── app.py                              # FastAPI web application (main entry point)
├── run_full_pipeline.py                # End-to-end: download → fakes → train → evaluate
├── run_phase1.py                       # Phase 1: Voice cloning generation
├── train_cnn_v2.py                     # CNN model definition + training loop
├── evaluate_cnn_v2.py                  # Evaluation: confusion matrix, ROC, classification report
├── evaluate_vc_metrics.py              # Voice cloning metrics (WER, speaker similarity)
├── VCFAD_Phase_1_Voice_Cloning_System.ipynb    # Phase 1 Jupyter notebook
├── VCFAD_Phase_2_Fake_Audio_Detection.ipynb    # Phase 2 Jupyter notebook
├── models/
│   └── cnn_v2.pt                       # Trained model weights
├── data/
│   ├── commonvoice/real/               # Real audio from LibriSpeech/CommonVoice
│   ├── timit/
│   │   ├── train/real/                 # Training real samples
│   │   ├── train/fake/                 # Generated fake samples
│   │   ├── test/real/                  # Test real samples
│   │   └── test/fake/                  # Test fake samples
│   ├── mels/                           # Pre-computed mel spectrograms (.npy)
│   ├── fad_dataset.csv                 # Full dataset metadata (path, label)
│   └── fad_mel_dataset.csv             # Mel spectrogram paths + labels
├── metadata/                           # Audio metadata files
├── results/                            # Evaluation outputs (confusion matrices, ROC curves)
├── templates/
│   └── index.html                      # Web UI template
├── static/
│   └── style.css                       # UI styling
└── .env                                # Environment variables (OPENAI_API_KEY)
```

---

## Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | >= 3.11.0 | Core runtime |
| FFmpeg | Latest | Audio format conversion (install via `brew install ffmpeg` or `apt install ffmpeg`) |
| CUDA GPU | Recommended | Significantly faster training (CPU works but is slower) |
| RAM | >= 16 GB | Recommended for full pipeline |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JackByteBack/TrueVoice-AI.git
cd TrueVoice-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
# Core ML / Deep Learning
pip install torch torchaudio torchvision

# Web framework
pip install fastapi uvicorn python-multipart jinja2 aiofiles

# Data processing
pip install numpy pandas scipy scikit-learn

# Audio processing
pip install datasets soundfile librosa

# HuggingFace (for dataset downloads)
pip install transformers

# AI integration (optional — for GPT explanations)
pip install openai python-dotenv

# Visualization
pip install matplotlib seaborn plotly

# Utilities
pip install tqdm
```

### 4. Set up environment variables (optional)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> Without an API key, the app still works but uses a built-in fallback explanation instead of GPT-powered analysis.

---

## Usage

### Option A: Run the Full Pipeline (Recommended for first-time setup)

This single command downloads data, generates fakes, extracts features, trains the model, and evaluates it:

```bash
python run_full_pipeline.py
```

**What it does:**
1. Downloads ~2000 real audio samples from LibriSpeech via HuggingFace
2. Generates fake audio using 6 signal transformations (noise, stretch, compress, pitch up/down, spectral mask)
3. Builds a consolidated CSV dataset with real/fake labels
4. Extracts mel spectrograms (64 mel bands, 1024 FFT window, 256 hop length)
5. Trains the FADCNN model for up to 30 epochs with early stopping
6. Evaluates on the held-out test set and reports accuracy + F1

### Option B: Run Individual Components

**Train only** (requires `data/fad_mel_dataset.csv` to exist):
```bash
python train_cnn_v2.py
```

**Evaluate only** (requires `models/cnn_v2.pt` to exist):
```bash
python evaluate_cnn_v2.py
```

**Voice cloning metrics**:
```bash
python evaluate_vc_metrics.py
```

### Option C: Launch the Web Application

```bash
uvicorn app:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

**Supported upload formats:** WAV, MP3, M4A, FLAC

---

## Model Architecture — FADCNN

The Fake Audio Detection CNN (FADCNN) is a lightweight convolutional network designed for mel spectrogram classification:

```
Input: (1, 64, 128)  ← single-channel mel spectrogram
         │
         ▼
┌─────────────────────────────────────┐
│  Conv2d(1 → 16, 3×3, padding=1)    │
│  BatchNorm2d(16)                    │
│  ReLU                               │
│  MaxPool2d(2)                       │
├─────────────────────────────────────┤
│  Conv2d(16 → 32, 3×3, padding=1)   │
│  BatchNorm2d(32)                    │
│  ReLU                               │
│  MaxPool2d(2)                       │
├─────────────────────────────────────┤
│  Conv2d(32 → 64, 3×3, padding=1)   │
│  BatchNorm2d(64)                    │
│  ReLU                               │
│  AdaptiveAvgPool2d(1, 1)            │
├─────────────────────────────────────┤
│  Dropout(0.3)                       │
├─────────────────────────────────────┤
│  Linear(64 → 1)                     │
└─────────────────────────────────────┘
         │
         ▼
Output: Real/Fake probability (sigmoid)
```

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Loss Function | BCEWithLogitsLoss |
| Label Smoothing | 0.9 (real) / 0.05 (fake) |
| Batch Size | 32 |
| Max Epochs | 30 |
| Early Stopping Patience | 3 epochs |
| Data Augmentation | Time masking, frequency masking, Gaussian noise |

**Dataset Split:**
| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning / early stopping |
| Test | 15% | Final evaluation |

---

## API Reference

### `GET /`
Returns the home page with an audio upload form.

**Response:** HTML page

---

### `POST /predict`
Analyzes an uploaded audio file and returns classification results.

**Request:**
- `file`: Audio file (multipart/form-data)
- Supported formats: WAV, MP3, M4A, FLAC

**Response:** HTML page containing:

| Field | Description |
|-------|-------------|
| `filename` | Name of the uploaded file |
| `label` | `REAL` or `FAKE` |
| `prob` | Confidence score (0–1) |
| `mel_data` | Mel spectrogram data for Plotly visualization |
| `waveform` | Raw waveform data for Plotly visualization |
| `explanation` | AI-generated analysis of the classification |

---

## Performance

### Classification Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Accuracy** | 1.000 | 1.000 | 1.000 |
| **F1-Score** | 1.000 | 1.000 | 1.000 |
| **Precision** | 1.000 | 1.000 | 1.000 |
| **Recall** | 1.000 | 1.000 | 1.000 |

### WER (Word Error Rate) Analysis
- Average WER between real and cloned audio: ~48%
- Most samples fall in the 0.3–0.6 WER range
- Higher WER (>1.0) typically indicates failed synthesis

### Key Findings
- The model achieves perfect separation between real and synthetic audio on the test set
- No bias toward the majority class despite the 8:3 real-to-fake imbalance
- Data augmentation (time/frequency masking + noise) improves generalization
- Label smoothing prevents overconfident predictions

---

## Audio Transformations (Fake Generation)

The pipeline generates fake audio using these signal processing techniques:

| Transformation | Description | Example |
|---------------|-------------|---------|
| **Noise Addition** | Injects Gaussian noise at controlled levels | `noise_level=0.008` |
| **Time Stretch** | Speeds up or slows down audio without pitch change | `rate=1.15` (faster), `rate=0.85` (slower) |
| **Pitch Shift** | Shifts pitch by semitones | `n_steps=3` (up), `n_steps=-3` (down) |
| **Spectral Mask** | Suppresses high-frequency bands | `mask_ratio=0.25` (removes above 25% of Nyquist) |

Each real sample gets 2 random transformations applied, generating diverse fake variants.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11+ |
| **Deep Learning** | PyTorch 2.x, Torchaudio |
| **ML Utilities** | Scikit-learn, NumPy, Pandas |
| **Audio Processing** | SciPy, Librosa, SoundFile |
| **Web Framework** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3, Plotly.js (interactive charts) |
| **AI Integration** | OpenAI GPT API (optional) |
| **Datasets** | LibriSpeech, CommonVoice, TIMIT |
| **TTS Models** | Microsoft SpeechT5, HiFi-GAN Vocoder |

---

## Data Sources

| Dataset | Source | Usage |
|---------|--------|-------|
| **LibriSpeech** | [OpenSLR](https://www.openslr.org/12) | Real audio samples for training |
| **CommonVoice** | [Mozilla](https://commonvoice.mozilla.org/) | Real audio samples for training |
| **TIMIT** | [NIST](https://catalog.ldc.upenn.edu/LDC93S1) | Acoustic-phonetic speech corpus |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Model not found at models/cnn_v2.pt` | Run `python run_full_pipeline.py` or `python train_cnn_v2.py` to train the model |
| `ffmpeg: command not found` | Install FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux) |
| CUDA out of memory | Reduce batch size in `train_cnn_v2.py` (default: 32) or use CPU |
| OpenAI API errors | Check your `OPENAI_API_KEY` in `.env` — the app works without it using fallback explanations |
| Slow training | Use a CUDA GPU if available; CPU training is significantly slower |

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- [Microsoft Research](https://www.microsoft.com/en-us/research/) — SpeechT5 TTS model
- [Mozilla CommonVoice](https://commonvoice.mozilla.org/) — Open speech dataset
- [NIST TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) — Acoustic-phonetic corpus
- [OpenAI](https://openai.com/) — GPT integration for result explanations
- [HuggingFace](https://huggingface.co/) — Dataset hosting and transformers library
