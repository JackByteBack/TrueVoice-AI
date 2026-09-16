from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch, torchaudio, numpy as np, os, io, base64, matplotlib.pyplot as plt, subprocess, tempfile
from scipy.io import wavfile as scipy_wavfile
from train_cnn_v2 import FADCNN
import seaborn as sns
sns.set_theme(style="darkgrid")  #  add this line
from dotenv import load_dotenv
from openai import OpenAI

#  Set your API key (better: load from env var)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key) if api_key else None
# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Fake Audio Detection Web App", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Preprocessing
# -----------------------------
def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".wav", ".wave"):
        return file_path
    wav_path = file_path.rsplit(".", 1)[0] + "_converted.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, check=True
    )
    return wav_path

def load_audio(file_path):
    wav_path = convert_to_wav(file_path)
    sr, data = scipy_wavfile.read(wav_path)
    if wav_path != file_path:
        os.remove(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

def preprocess_audio(file_path, target_sr=16000, n_mels=64, target_frames=128):
    wav_np, sr = load_audio(file_path)
    if sr != target_sr:
        tmp_wav = file_path + "_tmp16k.wav"
        subprocess.run(["ffmpeg", "-y", "-i", file_path, "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_wav], capture_output=True, check=True)
        sr, wav_np = scipy_wavfile.read(tmp_wav)
        os.remove(tmp_wav)
        if wav_np.dtype == np.int16:
            wav_np = wav_np.astype(np.float32) / 32768.0
        elif wav_np.dtype == np.int32:
            wav_np = wav_np.astype(np.float32) / 2147483648.0
    wav_tensor = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
    wav_tensor = wav_tensor / (wav_tensor.abs().max() + 1e-9)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=256, n_mels=n_mels
    )(wav_tensor)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel = mel_db.squeeze(0).numpy()
    n_mels, T = mel.shape
    if T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")
    elif T > target_frames:
        mel = mel[:, :target_frames]
    return mel.astype(np.float32)

#-----------------------------
# Extract Features
#-----------------------------
def extract_features(wav_tensor, sr):
    """
    wav_tensor: (T,) mono, normalized to [-1, 1]
    """
    wav = wav_tensor.cpu().numpy() if hasattr(wav_tensor, "cpu") else wav_tensor
    T = len(wav)
    dur = T / sr

    # Energy (RMS, dBFS-ish)
    rms = np.sqrt(np.mean(wav**2) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)

    # Silence ratio (below -40 dBFS relative)
    thr = (10 ** (-40/20))  # relative linear threshold
    silence_ratio = float(np.mean(np.abs(wav) < thr))

    # Short-time features via STFT
    spec = np.abs(np.fft.rfft(wav, n=2048))
    freqs = np.fft.rfftfreq(2048, d=1.0/sr)
    spec = spec + 1e-9
    spec_norm = spec / spec.sum()

    # Spectral centroid & bandwidth (simple global)
    centroid_hz = float((freqs * spec_norm).sum())
    spread = float(np.sqrt(((freqs - centroid_hz) ** 2 * spec_norm).sum()))

    # MFCC (with torchaudio)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64}
    )
    mfcc = mfcc_transform(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()  # (13, frames)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()
    mfcc_std  = np.std(mfcc, axis=1).tolist()

    return {
        "duration_sec": round(dur, 3),
        "rms_db": round(rms_db, 2),
        "silence_ratio": round(silence_ratio, 3),
        "spectral_centroid_hz": round(centroid_hz, 1),
        "spectral_spread_hz": round(spread, 1),
        "mfcc_mean": [round(v, 3) for v in mfcc_mean],
        "mfcc_std":  [round(v, 3) for v in mfcc_std],
    }

def build_fallback_explanation(features: dict, label: str, prob: float):
    dur = features.get("duration_sec", 0)
    rms = features.get("rms_db", -60)
    silence = features.get("silence_ratio", 0)
    centroid = features.get("spectral_centroid_hz", 0)
    parts = []
    if label == "FAKE":
        parts.append(f"This audio is classified as likely synthetic/fake with {prob*100:.1f}% confidence.")
    else:
        parts.append(f"This audio is classified as likely real/authentic with {(1-prob)*100:.1f}% confidence.")
    parts.append(f"The clip is {dur:.1f}s long with an average energy of {rms:.1f} dB.")
    if silence > 0.5:
        parts.append(f"High silence ratio ({silence*100:.0f}%) may indicate synthetic padding or heavy noise gating.")
    elif silence < 0.1:
        parts.append(f"Very low silence ratio ({silence*100:.0f}%) suggests continuous speech with minimal pauses.")
    if centroid > 3000:
        parts.append(f"The spectral centroid is high ({centroid:.0f} Hz), which can indicate sharper or more processed audio.")
    elif centroid < 1000:
        parts.append(f"The spectral centroid is low ({centroid:.0f} Hz), suggesting deeper or more muffled audio characteristics.")
    return " ".join(parts)

def summarize_with_gpt(features: dict, label: str, prob: float):
    if not client:
        return build_fallback_explanation(features, label, prob)
    prompt = f"""
You are helping explain a fake-audio detector's result to a user.
Given acoustic features and the model's prediction, write a concise 2–3 sentence summary
describing what the audio likely sounds like and why it may be classified as {label}.
Avoid jargon; keep it intuitive. Do not repeat the raw numbers verbatim—interpret them.

Prediction: {label} (confidence {prob:.3f})
Features (JSON):
{features}
"""
    try:
        resp = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You explain audio traits clearly to non-experts."},
                {"role": "user", "content": prompt},
            ],
            timeout=15,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return build_fallback_explanation(features, label, prob)

# -----------------------------
# Model loading
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/cnn_v2.pt"
model = FADCNN().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f" Model loaded from {model_path}")
else:
    print(f" WARNING: Model not found at {model_path}. Using uninitialized model.")

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})




#==============================BASELINE WORKING FUNCTION====================================#
# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, file: UploadFile = File(...)):
#     try:
#         # 1️⃣ Save uploaded file temporarily
#         tmp_path = f"temp_{file.filename}"
#         with open(tmp_path, "wb") as f:
#             f.write(await file.read())

#         # 2️⃣ Load waveform
#         wav, sr = torchaudio.load(tmp_path)
#         wav = wav.mean(dim=0)  # mono
#         wav = wav / (wav.abs().max() + 1e-9)
#         waveform = wav.cpu().numpy().tolist()

#         # 3️⃣ Preprocess to mel spectrogram
#         mel = preprocess_audio(tmp_path)
#         mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

#         # 4️⃣ Model inference
#         with torch.no_grad():
#             logits = model(mel_tensor)
#             prob = torch.sigmoid(logits).item()
#         label = "FAKE" if prob >= 0.5 else "REAL"

#         # 5️⃣ Clean up temp file
#         os.remove(tmp_path)

#         # 6️⃣ Render HTML with both waveform & mel spectrogram
#         return templates.TemplateResponse(
#             "index.html",
#             {
#                 "request": request,
#                 "result": {
#                     "filename": file.filename,
#                     "label": label,
#                     "prob": round(prob, 3),
#                     "mel_data": mel.tolist(),
#                     "waveform": waveform,
#                 },
#             },
#         )

#     except Exception as e:
#         return templates.TemplateResponse(
#             "index.html",
#             {"request": request, "error": str(e), "result": None},
#         )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        tmp_path = f"temp_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        # Load & normalize waveform
        wav_np, sr = load_audio(tmp_path)
        wav_tensor = torch.tensor(wav_np, dtype=torch.float32)
        waveform = wav_np.tolist()

        # Preprocess to mel for CNN
        mel = preprocess_audio(tmp_path)
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
        label = "FAKE" if prob >= 0.5 else "REAL"

        # === Smart explanation ===
        feats = extract_features(wav_tensor, sr)
        explanation = summarize_with_gpt(feats, label, prob)

        os.remove(tmp_path)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": {
                    "filename": file.filename,
                    "label": label,
                    "prob": round(prob, 3),
                    "mel_data": mel.tolist(),
                    "waveform": waveform,
                    "explanation": explanation,
                },
            },
        )

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "result": None},
        )


# # Run via:
# uvicorn app:app --reload --port 8000
