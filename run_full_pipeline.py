"""
Complete pipeline: Download data -> Generate fakes -> Extract mels -> Train CNN
Uses HuggingFace datasets for fast download + signal transformations for fakes
"""
import os, torch, numpy as np, pandas as pd, random, subprocess, time
from scipy.io import wavfile as scipy_wavfile
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "16"
torch.set_num_threads(16)
PROJECT = "/Users/jackobito/Desktop/2xqcTCqAYvy0SJbK-main"
os.chdir(PROJECT)

# ============================================================
# STEP 1: Download real audio from LibriSpeech via HuggingFace
# ============================================================
print("=" * 60)
print("STEP 1: Downloading real audio samples...")
print("=" * 60)

from datasets import load_dataset

print("Loading LibriSpeech test-clean (small subset)...")
ds = load_dataset("openslr/librispeech_asr", "clean", split="test", trust_remote_code=True)
print(f"Downloaded {len(ds)} samples")

# Save a subset as WAV files
N_REAL = 2000  # Use 2000 real samples
real_dir = "data/commonvoice/real"
os.makedirs(real_dir, exist_ok=True)

saved = 0
for i in tqdm(range(min(N_REAL * 3, len(ds))), desc="Saving real audio"):
    if saved >= N_REAL:
        break
    try:
        audio = ds[i]["audio"]["array"]
        sr = ds[i]["audio"]["sampling_rate"]
        # Convert to int16
        audio_np = np.array(audio, dtype=np.float32)
        peak = max(np.max(np.abs(audio_np)), 1e-9)
        audio_int16 = np.int16(audio_np / peak * 32767)
        fname = f"librispeech_{i:06d}.wav"
        scipy_wavfile.write(os.path.join(real_dir, fname), sr, audio_int16)
        saved += 1
    except:
        continue

print(f"Saved {saved} real audio files to {real_dir}")

# ============================================================
# STEP 2: Generate fake audio via signal transformations
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Generating fake audio via signal transformations...")
print("=" * 60)

import soundfile as sf
import scipy.signal

fake_dir = "data/timit/train/fake"
real_timit_dir = "data/timit/train/real"
os.makedirs(fake_dir, exist_ok=True)
os.makedirs(real_timit_dir, exist_ok=True)

real_files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]
print(f"Processing {len(real_files)} real files to create fakes...")

target_sr = 16000

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio)) * noise_level
    return np.clip(audio + noise, -1.0, 1.0)

def time_stretch(audio, sr, rate=1.1):
    indices = np.round(np.arange(0, len(audio), rate)).astype(int)
    indices = indices[indices < len(audio)]
    stretched = audio[indices]
    if len(stretched) > len(audio):
        stretched = stretched[:len(audio)]
    else:
        stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
    return stretched

def pitch_shift(audio, sr, n_steps=2):
    factor = 2 ** (n_steps / 12.0)
    indices = np.round(np.arange(0, len(audio), factor)).astype(int)
    indices = indices[indices < len(audio)]
    shifted = audio[indices]
    if len(shifted) > len(audio):
        shifted = shifted[:len(audio)]
    else:
        shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
    return shifted

def spectral_mask(audio, sr, mask_ratio=0.3):
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1.0/sr)
    mask = np.ones(len(spec))
    high_freq_idx = np.where(freqs > sr * mask_ratio)[0]
    mask[high_freq_idx] = 0.1
    spec = spec * mask
    return np.fft.irfft(spec, n=len(audio))

transforms = [
    ("noise", lambda a, sr: add_noise(a, 0.008)),
    ("stretch", lambda a, sr: time_stretch(a, sr, 1.15)),
    ("compress", lambda a, sr: time_stretch(a, sr, 0.85)),
    ("pitch_up", lambda a, sr: pitch_shift(a, sr, 3)),
    ("pitch_down", lambda a, sr: pitch_shift(a, sr, -3)),
    ("spectral", lambda a, sr: spectral_mask(a, sr, 0.25)),
]

saved_fake = 0
for fname in tqdm(real_files, desc="Generating fakes"):
    try:
        sr_orig, audio_int16 = scipy_wavfile.read(os.path.join(real_dir, fname))
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if sr_orig != target_sr:
            audio_float = scipy.signal.resample(audio_float, int(len(audio_float) * target_sr / sr_orig))

        # Pick 2 random transforms per file
        chosen = random.sample(transforms, min(2, len(transforms)))
        for tname, tfunc in chosen:
            fake_audio = tfunc(audio_float, target_sr)
            fake_audio = fake_audio.astype(np.float32)
            peak = max(np.max(np.abs(fake_audio)), 1e-9)
            fake_int16 = np.int16(fake_audio / peak * 32767)
            out_name = f"fake_{tname}_{fname}"
            scipy_wavfile.write(os.path.join(fake_dir, out_name), target_sr, fake_int16)
            saved_fake += 1
    except Exception as e:
        pass

print(f"Generated {saved_fake} fake audio files")

# Also copy some real files to timit train/real for dataset balance
for fname in tqdm(real_files[:1500], desc="Copying real to train"):
    try:
        sr_orig, data = scipy_wavfile.read(os.path.join(real_dir, fname))
        scipy_wavfile.write(os.path.join(real_timit_dir, fname), sr_orig, data)
    except:
        pass

# ============================================================
# STEP 3: Build dataset CSV
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Building dataset CSV...")
print("=" * 60)

from glob import glob

real_dirs = ["data/commonvoice/real", "data/timit/train/real"]
fake_dirs = ["data/timit/train/fake"]

real_paths = []
for d in real_dirs:
    real_paths.extend(glob(os.path.join(d, "*.wav")))
fake_paths = []
for d in fake_dirs:
    fake_paths.extend(glob(os.path.join(d, "*.wav")))

df_fad = pd.DataFrame({
    "path": real_paths + fake_paths,
    "label": [0] * len(real_paths) + [1] * len(fake_paths)
})
df_fad = df_fad.sample(frac=1, random_state=42).reset_index(drop=True)
df_fad["path"] = df_fad["path"].apply(lambda x: x.replace("\\", "/"))

os.makedirs("data", exist_ok=True)
df_fad.to_csv("data/fad_dataset.csv", index=False)
print(f"Dataset: {len(real_paths)} real + {len(fake_paths)} fake = {len(df_fad)} total")

# ============================================================
# STEP 4: Extract mel spectrograms
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Extracting mel spectrograms...")
print("=" * 60)

import torchaudio

mel_dir = Path("data/mels") if __import__('pathlib').Path else None
from pathlib import Path
mel_dir = Path("data/mels")
mel_dir.mkdir(parents=True, exist_ok=True)

mel_tx = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=256, n_mels=64
)
to_db = torchaudio.transforms.AmplitudeToDB()

paths, labels = [], []
for i, row in tqdm(df_fad.iterrows(), total=len(df_fad), desc="Computing mels"):
    try:
        sr_file, audio_int16 = scipy_wavfile.read(row.path)
        audio = audio_int16.astype(np.float32) / 32768.0
        if sr_file != 16000:
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr_file))
        wav = torch.tensor(audio).unsqueeze(0)
        mel = to_db(mel_tx(wav)).numpy()
        npy = mel_dir / f"{i:06d}.npy"
        np.save(npy, mel)
        paths.append(str(npy))
        labels.append(int(row.label))
    except Exception as e:
        pass

df_mel = pd.DataFrame({"path": paths, "label": labels})
df_mel.to_csv("data/fad_mel_dataset.csv", index=False)
print(f"Saved {len(paths)} mel spectrograms")
print(f"Label distribution: {dict(pd.Series(labels).value_counts())}")

# ============================================================
# STEP 5: Train the CNN
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training the CNN model...")
print("=" * 60)

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score

class MelDataset(Dataset):
    def __init__(self, df, target_frames=128, augment=True):
        self.df = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.cache = {}
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, label = self.df.iloc[idx]
        if path not in self.cache:
            self.cache[path] = np.load(path)
        mel = np.array(self.cache[path])
        if mel.ndim == 3:
            mel = mel.squeeze(0)
        elif mel.ndim == 1:
            mel = mel.reshape(64, -1)
        n_mels, T = mel.shape
        if T < self.target_frames:
            mel = np.pad(mel, ((0, 0), (0, self.target_frames - T)), mode="constant")
        elif T > self.target_frames:
            mel = mel[:, :self.target_frames]
        if self.augment:
            if random.random() < 0.3:
                t = random.randint(0, mel.shape[1] - 10)
                mel[:, t:t + 10] = 0
            if random.random() < 0.3:
                f = random.randint(0, mel.shape[0] - 8)
                mel[f:f + 8, :] = 0
            if random.random() < 0.3:
                mel = mel + 0.01 * np.random.randn(*mel.shape)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

from train_cnn_v2 import FADCNN

df_mel_train = pd.read_csv("data/fad_mel_dataset.csv")
ds_full = MelDataset(df_mel_train)
n = len(ds_full)
train_n = int(0.7 * n)
val_n = int(0.15 * n)
test_n = n - train_n - val_n
train_ds, val_ds, test_ds = random_split(ds_full, [train_n, val_n, test_n])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FADCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = torch.nn.BCEWithLogitsLoss()

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

best_f1, patience, counter = 0.0, 5, 0

def run_epoch(loader, train=False):
    model.train(mode=train)
    total_loss, preds, labels = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device).float(), yb.to(device).float()
        with torch.set_grad_enabled(train):
            logits = model(xb).squeeze(1)
            yb_smooth = yb * 0.9 + 0.05
            loss = crit(logits, yb_smooth)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds.extend((probs >= 0.5).astype(int))
        labels.extend(yb.cpu().numpy().astype(int))
    return total_loss / max(len(loader), 1), f1_score(labels, preds, zero_division=0)

print(f"Training on {train_n} samples, validating on {val_n}...")
start = time.time()
for epoch in range(30):
    tr_loss, tr_f1 = run_epoch(train_loader, True)
    va_loss, va_f1 = run_epoch(val_loader, False)
    print(f"Epoch {epoch+1:02d} | Train {tr_loss:.3f}/{tr_f1:.3f} | Val {va_loss:.3f}/{va_f1:.3f}")

    if va_f1 > best_f1:
        best_f1, counter = va_f1, 0
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/cnn_v2.pt")
        print("  -> Saved best model")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break

elapsed = (time.time() - start) / 60
print(f"\nTraining done in {elapsed:.1f} min. Best F1: {best_f1:.3f}")

# ============================================================
# STEP 6: Evaluate
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Final evaluation...")
print("=" * 60)

model.load_state_dict(torch.load("models/cnn_v2.pt", map_location=device))
model.eval()
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device).float()
        logits = model(xb).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend((probs >= 0.5).astype(int))
        all_labels.extend(yb.numpy().astype(int))

test_f1 = f1_score(all_labels, all_preds, zero_division=0)
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test Accuracy: {accuracy:.3f}")
print(f"Test F1: {test_f1:.3f}")
print(f"\nDone! Model saved at models/cnn_v2.pt")
