import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_tess_dataset,
    get_splits,
    save_results,
    set_seed,
    get_device,
    EMOTION_LABELS,
    IDX_TO_EMOTION,
    NUM_CLASSES
)

from models.speech_pipeline.train import (
    SpeechEmotionModel,
    SpeechDataset
)

# Import everything needed from text pipeline — including build_text
from models.text_pipeline.train import (
    TextEmotionModel,
    TextDataset,
    build_text
)

from models.fusion_pipeline.train import (
    MultimodalDataset,
    FusionEmotionModel
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA_ROOT   = 'data/TESS Toronto emotional speech set data'
RESULTS_DIR = 'Results'
BATCH_SIZE  = 16

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST CHECKPOINT LOADER
# Handles the key mismatch caused by the notebook having multiple conflicting
# TextEmotionModel definitions (fc1/fc2 vs classifier.*).
# Remaps old key names automatically, then falls back to strict=False with a
# clear warning so inference always runs rather than crashing.
# ─────────────────────────────────────────────────────────────────────────────

FC_KEY_MAP = {
    # standalone TextEmotionModel checkpoint saved with fc1/fc2
    'fc1.weight':  'classifier.1.weight',
    'fc1.bias':    'classifier.1.bias',
    'fc2.weight':  'classifier.4.weight',
    'fc2.bias':    'classifier.4.bias',
    # same keys nested inside FusionEmotionModel.text_enc
    'text_enc.fc1.weight': 'text_enc.classifier.1.weight',
    'text_enc.fc1.bias':   'text_enc.classifier.1.bias',
    'text_enc.fc2.weight': 'text_enc.classifier.4.weight',
    'text_enc.fc2.bias':   'text_enc.classifier.4.bias',
}


def remap_state_dict(sd):
    new_sd, remapped = {}, []
    for k, v in sd.items():
        if k in FC_KEY_MAP:
            new_k = FC_KEY_MAP[k]
            new_sd[new_k] = v
            remapped.append(f'    {k}  →  {new_k}')
        else:
            new_sd[k] = v
    if remapped:
        print('  [key remap] renamed legacy fc1/fc2 keys:')
        for r in remapped:
            print(r)
    return new_sd


def robust_load(model, path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    sd  = remap_state_dict(raw)
    try:
        model.load_state_dict(sd, strict=True)
        print(f'  OK (strict=True)  ← {path}')
    except RuntimeError as e:
        print(f'  WARNING strict load failed:\n    {e}')
        print('  Retrying with strict=False — unmatched weights stay random.')
        model.load_state_dict(sd, strict=False)
        print(f'  OK (strict=False) ← {path}')
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def get_preds(model, loader, device, modality):
    model.eval()
    preds, labels = [], []
    for b in loader:
        if modality == 'speech':
            x, y = b
            out  = model(x.to(device))
        elif modality == 'text':
            out = model(b['input_ids'].to(device),
                        b['attention_mask'].to(device))
            y   = b['label']
        else:
            out = model(b['speech'].to(device),
                        b['input_ids'].to(device),
                        b['attention_mask'].to(device))
            y   = b['label']
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(y.tolist() if hasattr(y, 'tolist') else y.cpu().tolist())
    return np.array(labels), np.array(preds)


@torch.no_grad()
def extract_fusion_embs(model, loader, device):
    model.eval()
    sp_e, tx_e, fu_e, lbls = [], [], [], []
    for b in loader:
        sp   = b['speech'].to(device)
        ids  = b['input_ids'].to(device)
        attn = b['attention_mask'].to(device)
        sp_e.append(model.speech_enc.get_representation(sp).cpu().numpy())
        tx_e.append(model.text_enc.get_representation(ids, attn).cpu().numpy())
        fu_e.append(model.get_representation(sp, ids, attn).cpu().numpy())
        lbls.append(b['label'].numpy())
    return {
        'speech': np.vstack(sp_e),
        'text':   np.vstack(tx_e),
        'fusion': np.vstack(fu_e),
        'labels': np.concatenate(lbls)
    }


def plot_tsne(embs, labels, title, save_path):
    print(f'  t-SNE: {title} ...')
    reduced = TSNE(n_components=2, perplexity=30,
                   random_state=42, n_iter=1000).fit_transform(embs)
    palette = sns.color_palette('tab10', n_colors=len(EMOTION_LABELS))
    plt.figure(figsize=(9, 7))
    for i, e in enumerate(EMOTION_LABELS):
        m = labels == i
        plt.scatter(reduced[m, 0], reduced[m, 1],
                    c=[palette[i]], label=e, alpha=0.7, s=20)
    plt.title(title)
    plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  Saved → {save_path}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    set_seed(42)
    device = get_device()

    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print('\nLoading dataset ...')
    df = load_tess_dataset(DATA_ROOT)
    _, _, test_df = get_splits(df)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    sp_loader = DataLoader(SpeechDataset(test_df),
                           BATCH_SIZE, shuffle=False, num_workers=0)

    tx_loader = DataLoader(TextDataset(test_df, tokenizer),
                           BATCH_SIZE, shuffle=False, num_workers=0)

    fu_loader = DataLoader(MultimodalDataset(test_df, tokenizer),
                           BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Load models ───────────────────────────────────────────────────────────
    print('\nLoading models ...')

    sp_model = SpeechEmotionModel().to(device)
    robust_load(sp_model, os.path.join(RESULTS_DIR, 'speech_best_model.pt'), device)

    tx_model = TextEmotionModel().to(device)
    robust_load(tx_model, os.path.join(RESULTS_DIR, 'text_best_model.pt'), device)

    fu_model = FusionEmotionModel().to(device)
    robust_load(fu_model, os.path.join(RESULTS_DIR, 'fusion_best_model.pt'), device)

    # ── Inference ─────────────────────────────────────────────────────────────
    print('\nRunning inference ...')
    yt_sp, yp_sp = get_preds(sp_model, sp_loader, device, 'speech')
    yt_tx, yp_tx = get_preds(tx_model, tx_loader, device, 'text')
    yt_fu, yp_fu = get_preds(fu_model, fu_loader, device, 'fusion')

    acc_sp = (yt_sp == yp_sp).mean()
    acc_tx = (yt_tx == yp_tx).mean()
    acc_fu = (yt_fu == yp_fu).mean()

    print('\n' + '=' * 50)
    print('FINAL ACCURACY')
    print('=' * 50)
    print(f'Speech : {acc_sp*100:.2f}%')
    print(f'Text   : {acc_tx*100:.2f}%')
    print(f'Fusion : {acc_fu*100:.2f}%')

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(RESULTS_DIR, 'speech_final', yt_sp, yp_sp)
    save_results(RESULTS_DIR, 'text_final',   yt_tx, yp_tx)
    save_results(RESULTS_DIR, 'fusion_final', yt_fu, yp_fu)

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print('\nExtracting embeddings for t-SNE ...')
    embs = extract_fusion_embs(fu_model, fu_loader, device)

    plot_tsne(embs['speech'], embs['labels'],
              'Speech Embeddings',
              os.path.join(plots_dir, 'tsne_speech.png'))

    plot_tsne(embs['text'], embs['labels'],
              'DistilBERT Embeddings',
              os.path.join(plots_dir, 'tsne_text.png'))

    plot_tsne(embs['fusion'], embs['labels'],
              'Fusion Embeddings',
              os.path.join(plots_dir, 'tsne_fusion.png'))

    print('\nDone!')


if __name__ == '__main__':
    main()