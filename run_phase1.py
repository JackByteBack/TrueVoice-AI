"""
Phase 1 - FAST VERSION: Just save real audio + generate clones
Skip VAD metadata and speaker embeddings (not needed for FAD training)
"""
import os, torch, numpy as np, soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "16"
torch.set_num_threads(16)
os.chdir("/Users/jackobito/Desktop/2xqcTCqAYvy0SJbK-main")

print("Loading TIMIT from deeplake...")
import deeplake
ds_train = deeplake.load("hub://activeloop/timit-train")
ds_test = deeplake.load("hub://activeloop/timit-test")
print(f"Train: {len(ds_train)}, Test: {len(ds_test)}")

print("\nLoading SpeechT5 for voice cloning...")
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
dummy_embeddings = [torch.randn(512) for _ in range(100)]

def process_split(ds, split_name, n_per_speaker=5):
    dir_real = f"data/timit/{split_name}/real"
    dir_fake = f"data/timit/{split_name}/fake"
    os.makedirs(dir_real, exist_ok=True)
    os.makedirs(dir_fake, exist_ok=True)

    existing_real = set(f.replace('.wav','') for f in os.listdir(dir_real) if f.endswith('.wav'))
    print(f"\n{split_name}: {len(existing_real)} real files already exist out of {len(ds)} total")

    # Group by speaker
    speaker_ids = {}
    for i in tqdm(range(len(ds)), desc=f"Indexing {split_name}"):
        try:
            spk = int(ds['speaker_ids'][i].numpy().item())
            if spk not in speaker_ids:
                speaker_ids[spk] = []
            speaker_ids[spk].append(i)
        except:
            continue

    print(f"  Found {len(speaker_ids)} unique speakers")

    total = 0
    for spk_id, indices in tqdm(speaker_ids.items(), desc=f"Cloning {split_name}"):
        chosen = indices[:n_per_speaker]
        for idx in chosen:
            fname = f"{spk_id}_{idx:04d}"
            if fname in existing_real:
                continue
            try:
                audio_np = np.asarray(ds['audios'][idx].numpy()).squeeze()
                text_val = ds['texts'][idx].data()
                text = text_val.decode('utf-8', errors='ignore') if isinstance(text_val, bytes) else str(text_val)
                text = text.strip()
                if not text:
                    text = "This is a placeholder sentence for speech synthesis."

                # Save real
                real_path = f"{dir_real}/{fname}.wav"
                peak = max(np.max(np.abs(audio_np)), 1e-9)
                audio_int16 = np.int16(audio_np / peak * 32767)
                sf.write(real_path, audio_int16, 16000, subtype="PCM_16")

                # Generate and save fake
                fake_path = f"{dir_fake}/{fname}.wav"
                inputs = processor(text=text, return_tensors="pt")
                speaker_emb = dummy_embeddings[hash(spk_id) % len(dummy_embeddings)].unsqueeze(0)
                speech = tts_model.generate_speech(inputs["input_ids"], speaker_emb, vocoder=vocoder)
                sf.write(fake_path, speech.numpy(), samplerate=16000, subtype="PCM_16")
                total += 1
            except Exception as e:
                pass

    rc = len([f for f in os.listdir(dir_real) if f.endswith('.wav')])
    fc = len([f for f in os.listdir(dir_fake) if f.endswith('.wav')])
    print(f"  {split_name}: Generated {total} new. Total Real={rc}, Fake={fc}")

process_split(ds_train, "train", 5)
process_split(ds_test, "test", 5)

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE!")
print("=" * 60)
