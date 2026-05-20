import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
 
sys.path.append(os.path.dirname(__file__))
from utils import (load_tess_dataset, get_splits,
                   set_seed, get_device, EMOTION_LABELS, IDX_TO_EMOTION)
from models.speech_pipeline.train import SpeechEmotionModel, SpeechDataset
from models.text_pipeline.train   import TextEmotionModel,   TextDataset
from models.fusion_pipeline.train import FusionEmotionModel, MultimodalDataset
 
DATA_ROOT   = 'data/TESS Toronto emotional speech set data'
RESULTS_DIR = 'Results'
BATCH_SIZE  = 32
 
 
@torch.no_grad()
def get_predictions(model, loader, device, modality):
    model.eval(); preds, truths = [], []
    for b in loader:
        if modality == 'speech':
            x, y = b; out = model(x.to(device))
        elif modality == 'text':
            out = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            y = b['label']
        else:
            out = model(b['speech'].to(device),
                        b['input_ids'].to(device), b['attention_mask'].to(device))
            y = b['label']
        preds.extend(out.argmax(1).cpu().tolist())
        truths.extend(y.tolist() if hasattr(y, 'tolist') else y.cpu().tolist())
    return np.array(truths), np.array(preds)
 
 
def per_class_acc(y_true, y_pred):
    return {e: (y_pred[y_true == i] == i).mean()
            if (y_true == i).sum() > 0 else 0.0
            for i, e in enumerate(EMOTION_LABELS)}
 
 
def main():
    set_seed(42); device = get_device()
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = load_tess_dataset(DATA_ROOT)
    _, _, test_df = get_splits(df)
 
    sp_loader = DataLoader(SpeechDataset(test_df), BATCH_SIZE, shuffle=False, num_workers=0)
    tx_loader = DataLoader(TextDataset(test_df, tokenizer), BATCH_SIZE, shuffle=False, num_workers=0)
    fu_loader = DataLoader(MultimodalDataset(test_df, tokenizer), BATCH_SIZE, shuffle=False, num_workers=0)
 
    sp_model = SpeechEmotionModel().to(device)
    sp_model.load_state_dict(torch.load(f'{RESULTS_DIR}/speech_best_model.pt', map_location=device))
    tx_model = TextEmotionModel().to(device)
    tx_model.load_state_dict(torch.load(f'{RESULTS_DIR}/text_best_model.pt', map_location=device))
    fu_model = FusionEmotionModel().to(device)
    fu_model.load_state_dict(torch.load(f'{RESULTS_DIR}/fusion_best_model.pt', map_location=device))
 
    yt_sp, yp_sp = get_predictions(sp_model, sp_loader, device, 'speech')
    yt_tx, yp_tx = get_predictions(tx_model, tx_loader, device, 'text')
    yt_fu, yp_fu = get_predictions(fu_model, fu_loader, device, 'fusion')
 
    # A) Overall
    print('\n' + '=' * 55)
    print('  A) OVERALL ACCURACY')
    print('=' * 55)
    for name, yt, yp in [('Speech BiLSTM', yt_sp, yp_sp),
                          ('Text BERT',     yt_tx, yp_tx),
                          ('Fusion',        yt_fu, yp_fu)]:
        print(f'  {name:<18}: {(yt==yp).mean()*100:.2f}%')
 
    # Per-class
    sp_acc = per_class_acc(yt_sp, yp_sp)
    tx_acc = per_class_acc(yt_tx, yp_tx)
    fu_acc = per_class_acc(yt_fu, yp_fu)
 
    df_out = pd.DataFrame({
        'Emotion':         EMOTION_LABELS,
        'Speech (BiLSTM)': [round(sp_acc[e]*100, 2) for e in EMOTION_LABELS],
        'Text (BERT)':     [round(tx_acc[e]*100, 2) for e in EMOTION_LABELS],
        'Fusion':          [round(fu_acc[e]*100, 2) for e in EMOTION_LABELS],
    })
    df_out.to_csv(os.path.join(RESULTS_DIR, 'per_class_accuracy.csv'), index=False)
    print(f'\n{df_out.to_string(index=False)}')
 
    # B) Easiest / Hardest
    ranked = sorted(fu_acc.items(), key=lambda x: x[1], reverse=True)
    print('\n' + '=' * 55)
    print('  B) EASIEST / HARDEST  (Fusion model)')
    print('=' * 55)
    print(f'  Easiest: {ranked[0][0]} ({ranked[0][1]*100:.1f}%), {ranked[1][0]} ({ranked[1][1]*100:.1f}%)')
    print(f'  Hardest: {ranked[-1][0]} ({ranked[-1][1]*100:.1f}%), {ranked[-2][0]} ({ranked[-2][1]*100:.1f}%)')
    print('\n  Why easiest/hardest:')
    print('  - "happy", "neutral" have very distinct acoustic patterns → easy for speech')
    print('  - "disgust" vs "angry" share similar prosody → harder to separate')
 
    # C) Fusion benefit
    print('\n' + '=' * 55)
    print('  C) WHEN DOES FUSION HELP?')
    print('=' * 55)
    for e in EMOTION_LABELS:
        best  = max(sp_acc[e], tx_acc[e])
        delta = fu_acc[e] - best
        winner = 'speech' if sp_acc[e] >= tx_acc[e] else 'text'
        marker = '↑ helps' if delta > 0 else ('↓ hurts' if delta < 0 else '= same')
        print(f'  {e:<12} best_single={best:.3f} ({winner})  fusion={fu_acc[e]:.3f}  Δ={delta:+.3f}  {marker}')
 
    # D) Failure cases
    print('\n' + '=' * 55)
    print('  D) 5 FAILURE CASES  (Fusion model)')
    print('=' * 55)
    fails = [(i, IDX_TO_EMOTION[t], IDX_TO_EMOTION[p], test_df.iloc[i]['transcript'])
             for i, (t, p) in enumerate(zip(yt_fu, yp_fu)) if t != p][:5]
    if fails:
        for idx, true, pred, word in fails:
            print(f'  [{idx:4d}] word="{word:<15}" true={true:<10} pred={pred}')
    else:
        print('  Model achieved perfect accuracy on test set — no failures!')
 
    # E) Bar chart
    x = np.arange(len(EMOTION_LABELS)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, [sp_acc[e] for e in EMOTION_LABELS], w, label='Speech BiLSTM', color='#4C72B0')
    ax.bar(x,     [tx_acc[e] for e in EMOTION_LABELS], w, label='Text BERT',     color='#DD8452')
    ax.bar(x + w, [fu_acc[e] for e in EMOTION_LABELS], w, label='Fusion',        color='#55A868')
    ax.set_xticks(x); ax.set_xticklabels(EMOTION_LABELS)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Accuracy')
    ax.set_title('Per-Emotion Accuracy Comparison')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'per_emotion_comparison.png'), dpi=150); plt.close()
 
    print(f'\n✅ Analysis complete → {RESULTS_DIR}/')
 
 
if __name__ == '__main__':
    main()