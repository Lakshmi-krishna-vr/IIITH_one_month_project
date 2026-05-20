"""
terminal_test.py — Terminal-based multi-pipeline test
Loads all trained models and directly predicts emotions from a local file 
and text string right inside your command-line interface.

Usage:
    python terminal_test.py --audio "path/to/audio.wav" --text "Your text here"
Or simply run without arguments to trigger interactive prompt:
    python terminal_test.py
"""

import os
import sys
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import DistilBertTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_device, set_seed,
    EMOTION_LABELS, NUM_CLASSES, IDX_TO_EMOTION
)
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train   import TextEmotionModel
from models.fusion_pipeline.train import FusionEmotionModel

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = 'Results'
SR          = 22050
DURATION    = 4.0
MAX_AUDIO   = 345
MAX_TEXT    = 32

EMOTION_EMOJI = {
    'angry':   '😡', 'disgust': '🤢', 'fear':    '😨',
    'happy':   '😊', 'neutral': '😐', 'ps':      '😲', 'sad': '😢'
}

FC_KEY_MAP = {
    'fc1.weight': 'classifier.1.weight', 'fc1.bias': 'classifier.1.bias',
    'fc2.weight': 'classifier.4.weight', 'fc2.bias': 'classifier.4.bias',
    'text_enc.fc1.weight': 'text_enc.classifier.1.weight',
    'text_enc.fc1.bias':   'text_enc.classifier.1.bias',
    'text_enc.fc2.weight': 'text_enc.classifier.4.weight',
    'text_enc.fc2.bias':   'text_enc.classifier.4.bias',
}

def remap_sd(sd):
    return {FC_KEY_MAP.get(k, k): v for k, v in sd.items()}

def robust_load(model, path, device):
    sd = remap_sd(torch.load(path, map_location=device, weights_only=False))
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# ── Audio Processing ──────────────────────────────────────────────────────────
def load_and_preprocess_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
    waveform, sr = sf.read(audio_path, dtype='float32')
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SR)
        sr = SR
        
    max_len = int(sr * DURATION)
    if len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)))
    else:
        waveform = waveform[:max_len]
        
    mfcc   = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40, n_fft=512, hop_length=256)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat   = np.vstack([mfcc, delta, delta2]).T   # (T, 120)
    
    if feat.shape[0] < MAX_AUDIO:
        feat = np.vstack([feat, np.zeros((MAX_AUDIO - feat.shape[0], 120), dtype=np.float32)])
    else:
        feat = feat[:MAX_AUDIO]
        
    mean = feat.mean(0, keepdims=True)
    std = feat.std(0, keepdims=True) + 1e-8
    return torch.tensor((feat - mean) / std, dtype=torch.float32).unsqueeze(0)

# ── Inference Workers ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_predictions(device, tokenizer, models, audio_path, text_input):
    sp_model, tx_model, fu_model = models
    
    # Process speech inputs
    feat = load_and_preprocess_audio(audio_path).to(device)
    
    # Process text inputs
    enc = tokenizer(text_input, max_length=MAX_TEXT, padding='max_length', truncation=True, return_tensors='pt')
    ids = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    
    # Run pipelines
    logits_sp = sp_model(feat)
    logits_tx = tx_model(ids, attn)
    logits_fu = fu_model(feat, ids, attn)
    
    probs_sp = torch.softmax(logits_sp, dim=1).squeeze().cpu().numpy()
    probs_tx = torch.softmax(logits_tx, dim=1).squeeze().cpu().numpy()
    probs_fu = torch.softmax(logits_fu, dim=1).squeeze().cpu().numpy()
    
    return {
        'speech': (int(probs_sp.argmax()), probs_sp),
        'text': (int(probs_tx.argmax()), probs_tx),
        'fusion': (int(probs_fu.argmax()), probs_fu)
    }

def print_result_block(title, idx, probs):
    label = EMOTION_LABELS[idx]
    emoji = EMOTION_EMOJI.get(label, '❓')
    conf = probs[idx] * 100
    print(f"\n=== {title.upper()} PIPELINE ===")
    print(f"Predicted Emotion : {label.upper()} {emoji}")
    print(f"Confidence        : {conf:.2f}%")
    print("Probability Breakdowns:")
    for i, emo in enumerate(EMOTION_LABELS):
        print(f"  └─ {emo:<8}: {probs[i]*100:>6.2f}%")

# ── Main Control ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Terminal test file runner for multi-modal inference.")
    parser.add_argument('--audio', type=str, help="Path to input .wav sound file")
    parser.add_argument('--text', type=str, help="Corresponding transcript or expression phrase")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Using Processing Unit Device: {device}")
    
    print("\n[1/2] Loading computational weights and dependencies...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    sp_model = SpeechEmotionModel().to(device)
    robust_load(sp_model, os.path.join(RESULTS_DIR, 'speech_best_model.pt'), device)
    
    tx_model = TextEmotionModel().to(device)
    robust_load(tx_model, os.path.join(RESULTS_DIR, 'text_best_model.pt'), device)
    
    fu_model = FusionEmotionModel().to(device)
    robust_load(fu_model, os.path.join(RESULTS_DIR, 'fusion_best_model.pt'), device)
    print("✓ All architecture files mapped successfully.")

    # Drop into interactive collection mode if flags are absent
    audio_path = args.audio
    text_input = args.text
    
    print("\n[2/2] Resolving data pipelines...")
    if not audio_path:
        audio_path = input("Enter path to your validation audio file (e.g., sample.wav): ").strip()
    if not text_input:
        text_input = input("Enter phrase textual complement: ").strip()

    try:
        models = (sp_model, tx_model, fu_model)
        results = run_predictions(device, tokenizer, models, audio_path, text_input)
        
        print("\n" + "═"*60)
        print("                 EVALUATION LOG COMPLETED                     ")
        print("═"*60)
        print(f"Target Audio : {audio_path}")
        print(f"Target Text  : \"{text_input}\"")
        
        print_result_block("Speech-Only", *results['speech'])
        print_result_block("Text-Only", *results['text'])
        print_result_block("Multimodal Fusion", *results['fusion'])
        print("\n" + "═"*60)
        
    except Exception as e:
        print(f"\n❌ Operational failure occurred processing data inputs.")
        print(f"Error Context: {str(e)}")

if __name__ == '__main__':
    main()