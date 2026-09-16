# Voice Cloning & Fake Audio Detection System

A comprehensive end-to-end system for voice cloning and fake audio detection using deep learning and signal processing techniques.

**YouTube Presentation**: [Watch Demo](https://www.youtube.com/watch?v=RAfuJb4cujo)

---

## Overview

This project implements a complete pipeline for:
- **Voice Cloning**: Generating synthetic voice clones from real audio samples using SpeechT5 TTS
- **Fake Audio Detection**: Training a CNN-based classifier to distinguish between real and synthetic audio
- **Web Application**: FastAPI-based interface for real-time audio analysis and classification

---

## Project Structure

```
.
├── app.py                          # FastAPI web application
├── run_full_pipeline.py            # Complete end-to-end pipeline
├── run_phase1.py                   # Phase 1: Voice cloning generation
├── train_cnn_v2.py                 # CNN model definition and training
├── evaluate_cnn_v2.py              # Model evaluation and metrics
├── evaluate_vc_metrics.py          # Voice cloning metrics (WER, etc.)
├── VCFAD_Phase_1_Voice_Cloning_System.ipynb    # Phase 1 notebook
├── VCFAD_Phase_2_Fake_Audio_Detection.ipynb    # Phase 2 notebook
├── models/                         # Saved model weights
│   └── cnn_v2.pt
├── data/                           # Audio datasets
│   ├── commonvoice/real/           # CommonVoice real samples
│   ├── timit/                      # TIMIT dataset
│   │   ├── train/real/
│   │   ├── train/fake/
│   │   ├── test/real/
│   │   └── test/fake/
│   ├── mels/                       # Extracted mel spectrograms
│   ├── fad_dataset.csv             # Dataset metadata
│   └── fad_mel_dataset.csv         # Mel spectrogram metadata
├── metadata/                       # Audio metadata files
├── results/                        # Evaluation results
├── templates/                      # HTML templates
│   └── index.html
├── static/                         # Static assets
│   └── style.css
└── .env                            # Environment variables
```

---

## Prerequisites

- Python >= 3.11.0
- FFmpeg (for audio format conversion)
- CUDA-compatible GPU (recommended for training)
- Minimum 16GB RAM recommended

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 2xqcTCqAYvy0SJbK-main
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchaudio torchvision
   pip install fastapi uvicorn python-multipart
   pip install numpy pandas scipy scikit-learn
   pip install datasets soundfile librosa
   pip install transformers deeplake
   pip install openai python-dotenv
   pip install matplotlib seaborn plotly
   pip install jinja2 aiofiles
   pip install tqdm
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## Usage

### Option 1: Run the Complete Pipeline

Execute the full pipeline from data download to model training:

```bash
python run_full_pipeline.py
```

This will:
1. Download real audio samples from LibriSpeech
2. Generate fake audio via signal transformations
3. Build dataset CSV
4. Extract mel spectrograms
5. Train the CNN model
6. Evaluate performance

### Option 2: Run Individual Phases

**Phase 1 - Voice Cloning**:
```bash
python run_phase1.py
```

**Train the CNN Model**:
```bash
python train_cnn_v2.py
```

**Evaluate the Model**:
```bash
python evaluate_cnn_v2.py
```

### Option 3: Run the Web Application

```bash
uvicorn app:app --reload --port 8000
```

Then open your browser and navigate to `http://localhost:8000`

---

## Pipeline Architecture

### Phase 1: Voice Cloning System (VCS)

| Component | Description |
|-----------|-------------|
| **Input** | Real audio samples from TIMIT dataset |
| **Model** | Microsoft SpeechT5 TTS + HiFi-GAN Vocoder |
| **Output** | Synthetic voice clones (5 per speaker) |
| **Datasets** | TIMIT, CommonVoice |

### Phase 2: Signal Processing & Dataset Consolidation

| Component | Description |
|-----------|-------------|
| **Ground Truth** | CommonVoice dataset |
| **Transformations** | Noise addition, time stretching, pitch shifting, spectral masking |
| **Feature Extraction** | Mel spectrograms (64 mel bands, 1024 FFT, 256 hop length) |
| **Output** | Consolidated dataset with real/fake labels |

### Phase 3: Fake Audio Detection (FAD)

| Component | Description |
|-----------|-------------|
| **Model** | Custom CNN (FADCNN) |
| **Architecture** | 3 Conv2D layers + BatchNorm + AdaptiveAvgPool + FC |
| **Loss** | BCEWithLogitsLoss with label smoothing |
| **Optimizer** | Adam (lr=1e-3, weight_decay=1e-4) |
| **Metrics** | F1-Score, Accuracy, WER, Speaker Classification |

### Phase 4: Deployment

| Component | Description |
|-----------|-------------|
| **Backend** | FastAPI |
| **Frontend** | HTML5 + CSS + Plotly.js |
| **Server** | Uvicorn + NGINX |
| **AI Integration** | OpenAI GPT-3.5 for result explanations |

---

## Model Architecture (FADCNN)

```
Input: (1, 64, 128) mel spectrogram
    ↓
Conv2d(1→16, 3×3) + BatchNorm + ReLU + MaxPool(2)
    ↓
Conv2d(16→32, 3×3) + BatchNorm + ReLU + MaxPool(2)
    ↓
Conv2d(32→64, 3×3) + BatchNorm + ReLU + AdaptiveAvgPool(1,1)
    ↓
Dropout(0.3)
    ↓
Linear(64→1)
    ↓
Output: Real/Fake probability
```

---

## API Endpoints

### `GET /`
- **Description**: Home page with upload form
- **Response**: HTML page

### `POST /predict`
- **Description**: Analyze uploaded audio file
- **Request**: Multipart form data with audio file
- **Response**: HTML page with results including:
  - Classification (REAL/FAKE)
  - Confidence probability
  - Mel spectrogram visualization
  - Waveform plot
  - AI-generated explanation

---

## Performance Metrics

### Classification Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Accuracy** | 1.000 | 1.000 | 1.000 |
| **F1-Score** | 1.000 | 1.000 | 1.000 |
| **Precision** | 1.000 | 1.000 | 1.000 |
| **Recall** | 1.000 | 1.000 | 1.000 |

### WER Analysis
- Average WER between real and cloned audio: ~48%
- Most samples have WER between 0.3-0.6
- Higher WER (>1.0) indicates failed synthesis

### Key Observations
- Model achieves perfect separation between real and fake audio
- No bias toward majority class despite 8:3 imbalance
- WER and classification metrics reinforce each other as validation signals

---

## Technologies Used

| Category | Technologies |
|----------|--------------|
| **Languages** | Python 3.11+ |
| **Deep Learning** | PyTorch, Torchaudio |
| **ML Libraries** | Scikit-learn, NumPy, Pandas |
| **Audio Processing** | SciPy, Librosa, SoundFile |
| **Web Framework** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS, Plotly.js |
| **AI Integration** | OpenAI API |
| **Datasets** | TIMIT, CommonVoice, LibriSpeech |
| **TTS Models** | Microsoft SpeechT5, HiFi-GAN |

---

## Data Sources

- **TIMIT**: Acoustic-phonetic speech corpus (630 speakers, 8 dialect regions)
- **CommonVoice**: Mozilla's crowdsourced speech dataset
- **LibriSpeech**: Large-scale ASR corpus (used for real audio samples)

---

## Important Notes

1. **Model Location**: Ensure `models/cnn_v2.pt` exists before running the web app
2. **FFmpeg Required**: The system uses FFmpeg for audio format conversion
3. **GPU Recommended**: Training is significantly faster with CUDA support
4. **API Key**: OpenAI API key is optional but enables AI-generated explanations

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- Microsoft Research for SpeechT5 model
- Mozilla CommonVoice project
- TIMIT dataset by NIST
- OpenAI for GPT integration
