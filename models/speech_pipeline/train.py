import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import (load_tess_dataset, get_splits, load_audio, extract_mfcc,
                   save_results, set_seed, get_device, NUM_CLASSES)
 
 
# ── Dataset ────────────────────────────────────────────────────────────────
 
class SpeechDataset(Dataset):
    def __init__(self, df, sr=22050, duration=4.0, max_len=345):
        self.df = df; self.sr = sr; self.duration = duration; self.max_len = max_len
 
    def __len__(self): return len(self.df)
 
    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        waveform, sr = load_audio(row['file_path'], sr=self.sr, duration=self.duration)
        features = extract_mfcc(waveform, sr=sr)
        T = features.shape[0]
        if T < self.max_len:
            features = np.vstack([features,
                                  np.zeros((self.max_len - T, features.shape[1]),
                                           dtype=np.float32)])
        else:
            features = features[:self.max_len]
        mean = features.mean(0, keepdims=True)
        std  = features.std(0,  keepdims=True) + 1e-8
        return (torch.tensor((features - mean) / std, dtype=torch.float32),
                torch.tensor(row['label'], dtype=torch.long))
 
 
# ── Model ──────────────────────────────────────────────────────────────────
 
class SpeechEmotionModel(nn.Module):
    def __init__(self, input_size=120, hidden_size=256,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
 
    def forward(self, x):
        out, _ = self.bilstm(x)
        w      = torch.softmax(self.attention(out), dim=1)
        ctx    = (w * out).sum(dim=1)
        return self.classifier(ctx)
 
    def get_representation(self, x):
        out, _ = self.bilstm(x)
        w      = torch.softmax(self.attention(out), dim=1)
        return (w * out).sum(dim=1)
 
 
# ── Loops ──────────────────────────────────────────────────────────────────
 
def train_epoch(model, loader, opt, crit, device):
    model.train(); loss_sum = correct = total = 0
    for x, y in tqdm(loader, desc='train', leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x); loss = crit(logits, y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total
 
 
@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval(); loss_sum = correct = total = 0; preds = []; labels = []
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x); loss = crit(logits, y)
        loss_sum += loss.item() * y.size(0); p = logits.argmax(1)
        correct  += (p == y).sum().item(); total += y.size(0)
        preds.extend(p.cpu().tolist()); labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, preds, labels
 
 
# ── Train function ─────────────────────────────────────────────────────────
 
def train(data_root, epochs=30, batch_size=32, lr=1e-3, results_dir='Results'):
    set_seed(42); device = get_device()
    df = load_tess_dataset(data_root); tr, va, te = get_splits(df)
    trl = DataLoader(SpeechDataset(tr), batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    vl  = DataLoader(SpeechDataset(va), batch_size, shuffle=False, num_workers=2)
    tel = DataLoader(SpeechDataset(te), batch_size, shuffle=False, num_workers=2)
    model = SpeechEmotionModel().to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5, factor=0.5)
    hist  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best  = 0.0; spath = os.path.join(results_dir, 'speech_best_model.pt')
    os.makedirs(results_dir, exist_ok=True)
    print('\n=== SPEECH TRAINING ===')
    for ep in range(1, epochs + 1):
        tl, ta   = train_epoch(model, trl, opt, crit, device)
        vl_, va_, _, _ = evaluate(model, vl, crit, device); sched.step(va_)
        hist['train_loss'].append(tl); hist['val_loss'].append(vl_)
        hist['train_acc'].append(ta);  hist['val_acc'].append(va_)
        print(f'Ep {ep:03d}/{epochs} | TrLoss {tl:.4f} TrAcc {ta:.4f} | VaLoss {vl_:.4f} VaAcc {va_:.4f}')
        if va_ > best:
            best = va_; torch.save(model.state_dict(), spath)
            print(f'  Saved best ({best:.4f})')
    model.load_state_dict(torch.load(spath, map_location=device))
    _, acc, yp, yt = evaluate(model, tel, crit, device)
    print(f'Test Accuracy: {acc:.4f}')
    save_results(results_dir, 'speech', yt, yp, hist)
    return model
 
 
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   type=str, required=True)
    p.add_argument('--epochs',      type=int, default=30)
    p.add_argument('--results_dir', type=str, default='Results')
    a = p.parse_args(); train(a.data_root, a.epochs, results_dir=a.results_dir)