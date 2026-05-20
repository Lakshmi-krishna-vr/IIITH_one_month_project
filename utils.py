import os, re, json
import numpy as np
import pandas as pd
import librosa
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
 
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
NUM_CLASSES    = len(EMOTION_LABELS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}
 
def load_tess_dataset(data_root: str):
    records = []
    data_root = Path(data_root)
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir(): continue
        for wav_file in sorted(folder.glob('*.wav')):
            name  = wav_file.stem.lower()
            parts = name.split('_')
            emotion = parts[-1]
            word    = '_'.join(parts[1:-1])
            if emotion in ('pleasantsurprise', 'pleasantssurprise'):
                emotion = 'ps'
            if emotion not in EMOTION_TO_IDX: continue
            records.append({'file_path': str(wav_file), 'emotion': emotion,
                            'transcript': word, 'label': EMOTION_TO_IDX[emotion]})
    df = pd.DataFrame(records)
    print(f'Loaded {len(df)} samples | {df["emotion"].value_counts().to_dict()}')
    return df
 
def get_splits(df, test_size=0.15, val_size=0.15, seed=42):
    train_val, test = train_test_split(df, test_size=test_size,
                                        stratify=df['label'], random_state=seed)
    rv = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rv,
                                   stratify=train_val['label'], random_state=seed)
    print(f'Split → train={len(train)} val={len(val)} test={len(test)}')
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
 
def load_audio(path, sr=22050, duration=4.0):
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    max_len = int(sr * duration)
    if len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)))
    else:
        waveform = waveform[:max_len]
    return waveform, sr
 
def extract_mfcc(waveform, sr=22050, n_mfcc=40, n_fft=512, hop_length=256):
    mfcc   = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop_length)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T   # (T, 120)
 
def save_results(results_dir, model_name, y_true, y_pred, history=None):
    os.makedirs(results_dir, exist_ok=True)
    report    = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_dir, f'{model_name}_accuracy_table.csv'))
    print(f'\n=== {model_name} ===')
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title(f'{model_name} Confusion Matrix'); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_confusion_matrix.png'), dpi=150)
    plt.close()
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label='Train')
        axes[0].plot(epochs, history['val_loss'],   label='Val')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[1].plot(epochs, history['train_acc'],  label='Train')
        axes[1].plot(epochs, history['val_acc'],    label='Val')
        axes[1].set_title('Accuracy'); axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{model_name}_history.png'), dpi=150)
        plt.close()
    return report_df
 
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 
def get_device():
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {d}')
    return d