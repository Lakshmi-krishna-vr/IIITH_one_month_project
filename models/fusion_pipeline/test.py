import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.manifold import TSNE
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import (load_tess_dataset, get_splits, save_results,
                   set_seed, get_device, EMOTION_LABELS)
from models.fusion_pipeline.train import FusionEmotionModel, MultimodalDataset
 
 
@torch.no_grad()
def extract_embeddings(model, loader, device):
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
    return {'speech': np.vstack(sp_e), 'text': np.vstack(tx_e),
            'fusion': np.vstack(fu_e), 'labels': np.concatenate(lbls)}
 
 
def plot_tsne(embs, labels, title, save_path):
    print(f'  t-SNE: {title} …')
    reduced = TSNE(n_components=2, perplexity=30, random_state=42,
                   n_iter=1000).fit_transform(embs)
    palette = sns.color_palette('tab10', n_colors=len(EMOTION_LABELS))
    plt.figure(figsize=(9, 7))
    for i, e in enumerate(EMOTION_LABELS):
        m = labels == i
        plt.scatter(reduced[m, 0], reduced[m, 1], c=[palette[i]],
                    label=e, alpha=0.7, s=20)
    plt.title(title); plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f'  Saved → {save_path}')
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   type=str, default='data/TESS Toronto emotional speech set data')
    p.add_argument('--model_path',  type=str, default='Results/fusion_best_model.pt')
    p.add_argument('--results_dir', type=str, default='Results')
    p.add_argument('--batch_size',  type=int, default=16)
    args = p.parse_args()
 
    set_seed(42); device = get_device()
    plots_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
 
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    df  = load_tess_dataset(args.data_root)
    _, _, test_df = get_splits(df)
    loader = DataLoader(MultimodalDataset(test_df, tok), args.batch_size,
                        shuffle=False, num_workers=0)
 
    model = FusionEmotionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f'Loaded fusion model from {args.model_path}')
 
    # Accuracy
    all_preds, all_labels = [], []
    with torch.no_grad():
        for b in loader:
            sp, ids, attn, y = (b['speech'].to(device), b['input_ids'].to(device),
                                b['attention_mask'].to(device), b['label'].to(device))
            preds = model(sp, ids, attn).argmax(1)
            all_preds.extend(preds.cpu().tolist()); all_labels.extend(y.cpu().tolist())
 
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f'\nFusion Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)')
    save_results(args.results_dir, 'fusion_test', all_labels, all_preds)
 
    # t-SNE
    print('\nExtracting embeddings …')
    embs = extract_embeddings(model, loader, device)
    plot_tsne(embs['speech'], embs['labels'],
              'Temporal Modelling Block – Speech BiLSTM',
              os.path.join(plots_dir, 'tsne_temporal.png'))
    plot_tsne(embs['text'], embs['labels'],
              'Contextual Modelling Block – BERT',
              os.path.join(plots_dir, 'tsne_contextual.png'))
    plot_tsne(embs['fusion'], embs['labels'],
              'Fusion Block – Gated Multimodal',
              os.path.join(plots_dir, 'tsne_fusion.png'))
    print(f'\nAll t-SNE plots saved to {plots_dir}/')
 
 
if __name__ == '__main__':
    main()