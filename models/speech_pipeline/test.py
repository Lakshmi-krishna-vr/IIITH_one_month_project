import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import (load_tess_dataset, get_splits, load_audio, extract_mfcc,
                   save_results, set_seed, get_device, IDX_TO_EMOTION, EMOTION_LABELS)
from models.speech_pipeline.train import SpeechEmotionModel, SpeechDataset
 
 
def predict_wav(wav_path, model, device, max_len=345):
    waveform, sr = load_audio(wav_path)
    features     = extract_mfcc(waveform, sr=sr)
    T = features.shape[0]
    if T < max_len:
        features = np.vstack([features, np.zeros((max_len - T, features.shape[1]), dtype=np.float32)])
    else:
        features = features[:max_len]
    mean = features.mean(0, keepdims=True); std = features.std(0, keepdims=True) + 1e-8
    features = (features - mean) / std
    tensor   = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    pred = probs.argmax()
    print(f'\nPredicted emotion : {IDX_TO_EMOTION[pred]}')
    for i, p in enumerate(probs):
        print(f'  {IDX_TO_EMOTION[i]:<12} {p:.4f}  {"█" * int(p * 30)}')
    return IDX_TO_EMOTION[pred]
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   type=str, default='data/TESS Toronto emotional speech set data')
    p.add_argument('--model_path',  type=str, default='Results/speech_best_model.pt')
    p.add_argument('--results_dir', type=str, default='Results')
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--wav_file',    type=str, default=None)
    args = p.parse_args()
 
    set_seed(42); device = get_device()
    model = SpeechEmotionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f'Loaded model from {args.model_path}')
 
    if args.wav_file:
        predict_wav(args.wav_file, model, device); return
 
    df = load_tess_dataset(args.data_root)
    _, _, test_df = get_splits(df)
    loader = DataLoader(SpeechDataset(test_df), args.batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1)
            all_preds.extend(preds.cpu().tolist()); all_labels.extend(y.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f'\nTest Accuracy: {acc:.4f}  ({acc*100:.2f}%)')
    save_results(args.results_dir, 'speech_test', all_labels, all_preds)
 
 
if __name__ == '__main__':
    main()