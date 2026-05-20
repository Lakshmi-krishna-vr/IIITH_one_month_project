import os
import sys
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer

from tqdm import tqdm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECT PATH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..'
    )
)

from utils import (

    load_tess_dataset,
    get_splits,
    load_audio,
    extract_mfcc,
    save_results,
    set_seed,
    get_device,
    NUM_CLASSES
)

from models.speech_pipeline.train import (
    SpeechEmotionModel
)

from models.text_pipeline.train import (
    TextEmotionModel
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTIMODAL DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultimodalDataset(Dataset):

    def __init__(
        self,
        df,
        tokenizer,
        sr=22050,
        duration=4.0,
        max_len_audio=345,
        max_len_text=64
    ):

        self.df = df

        self.tok = tokenizer

        self.sr = sr

        self.duration = duration

        self.mla = max_len_audio

        self.mlt = max_len_text


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ━━━━━━━━━━━━━━━━━━━━━
        # LOAD AUDIO
        # ━━━━━━━━━━━━━━━━━━━━━

        waveform, sr = load_audio(

            row['file_path'],

            sr=self.sr,

            duration=self.duration
        )

        feat = extract_mfcc(

            waveform,

            sr=sr
        )

        T = feat.shape[0]

        if T < self.mla:

            feat = np.vstack([

                feat,

                np.zeros(

                    (self.mla - T, feat.shape[1]),

                    dtype=np.float32
                )
            ])

        else:

            feat = feat[:self.mla]

        # NORMALIZE

        mean = feat.mean(0, keepdims=True)

        std = feat.std(0, keepdims=True) + 1e-8

        feat = (feat - mean) / std


        # ━━━━━━━━━━━━━━━━━━━━━
        # TOKENIZE TEXT
        # ━━━━━━━━━━━━━━━━━━━━━

        enc = self.tok(

            row['transcript'].lower(),

            max_length=self.mlt,

            padding='max_length',

            truncation=True,

            return_tensors='pt'
        )

        return {

            'speech':

                torch.tensor(
                    feat,
                    dtype=torch.float32
                ),

            'input_ids':

                enc['input_ids'].squeeze(0),

            'attention_mask':

                enc['attention_mask'].squeeze(0),

            'label':

                torch.tensor(
                    row['label'],
                    dtype=torch.long
                )
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GATED FUSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GatedFusion(nn.Module):

    def __init__(

        self,

        sd=512,

        td=768,

        fd=512
    ):

        super().__init__()

        self.ps = nn.Linear(sd, fd)

        self.pt = nn.Linear(td, fd)

        self.gate = nn.Linear(
            sd + td,
            fd
        )

    def forward(self, s, t):

        sp = torch.tanh(
            self.ps(s)
        )

        tp = torch.tanh(
            self.pt(t)
        )

        g = torch.sigmoid(

            self.gate(

                torch.cat([s, t], dim=1)
            )
        )

        return g * sp + (1 - g) * tp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUSION MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FusionEmotionModel(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        dropout=0.3
    ):

        super().__init__()

        self.speech_enc = SpeechEmotionModel()

        self.text_enc = TextEmotionModel()

        # REMOVE CLASSIFIERS

        self.speech_enc.classifier = nn.Identity()

        self.text_enc.classifier = nn.Identity()

        self.fusion = GatedFusion()

        self.classifier = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(

        self,

        speech,

        input_ids,

        attention_mask
    ):

        s = self.speech_enc.get_representation(
            speech
        )

        t = self.text_enc.get_representation(
            input_ids,
            attention_mask
        )

        fused = self.fusion(s, t)

        return self.classifier(fused)

    def get_representation(

        self,

        speech,

        input_ids,

        attention_mask
    ):

        s = self.speech_enc.get_representation(
            speech
        )

        t = self.text_enc.get_representation(
            input_ids,
            attention_mask
        )

        return self.fusion(s, t)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN EPOCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_epoch(

    model,

    loader,

    opt,

    crit,

    device
):

    model.train()

    loss_sum = 0

    correct = 0

    total = 0

    for b in tqdm(loader, desc='train', leave=False):

        sp = b['speech'].to(device)

        ids = b['input_ids'].to(device)

        attn = b['attention_mask'].to(device)

        y = b['label'].to(device)

        opt.zero_grad()

        logits = model(
            sp,
            ids,
            attn
        )

        loss = crit(logits, y)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        opt.step()

        loss_sum += loss.item() * y.size(0)

        correct += (
            logits.argmax(1) == y
        ).sum().item()

        total += y.size(0)

    return loss_sum / total, correct / total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()

def evaluate(

    model,

    loader,

    crit,

    device
):

    model.eval()

    loss_sum = 0

    correct = 0

    total = 0

    preds = []

    labels = []

    for b in loader:

        sp = b['speech'].to(device)

        ids = b['input_ids'].to(device)

        attn = b['attention_mask'].to(device)

        y = b['label'].to(device)

        logits = model(
            sp,
            ids,
            attn
        )

        loss = crit(logits, y)

        loss_sum += loss.item() * y.size(0)

        p = logits.argmax(1)

        correct += (p == y).sum().item()

        total += y.size(0)

        preds.extend(
            p.cpu().tolist()
        )

        labels.extend(
            y.cpu().tolist()
        )

    return (

        loss_sum / total,

        correct / total,

        preds,

        labels
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(

    data_root,

    speech_ckpt='Results/speech_best_model.pt',

    text_ckpt='Results/text_best_model.pt',

    epochs=30,

    batch_size=8,

    lr=2e-5,

    weight_decay=0.01,

    results_dir='Results',

    save_best=True,

    early_stopping_patience=5
):

    set_seed(42)

    device = get_device()

    # ━━━━━━━━━━━━━━━━━━━━━
    # DISTILBERT TOKENIZER
    # ━━━━━━━━━━━━━━━━━━━━━

    tok = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased'
    )

    # ━━━━━━━━━━━━━━━━━━━━━
    # DATA
    # ━━━━━━━━━━━━━━━━━━━━━

    df = load_tess_dataset(data_root)

    tr, va, te = get_splits(df)

    trl = DataLoader(

        MultimodalDataset(tr, tok),

        batch_size=batch_size,

        shuffle=True,

        num_workers=2,

        pin_memory=True
    )

    vl = DataLoader(

        MultimodalDataset(va, tok),

        batch_size=batch_size,

        shuffle=False,

        num_workers=2
    )

    tel = DataLoader(

        MultimodalDataset(te, tok),

        batch_size=batch_size,

        shuffle=False,

        num_workers=2
    )

    # ━━━━━━━━━━━━━━━━━━━━━
    # MODEL
    # ━━━━━━━━━━━━━━━━━━━━━

    model = FusionEmotionModel().to(device)

    # ━━━━━━━━━━━━━━━━━━━━━
    # LOAD SPEECH MODEL
    # ━━━━━━━━━━━━━━━━━━━━━

    if os.path.exists(speech_ckpt):

        st = {

            k: v

            for k, v in torch.load(

                speech_ckpt,

                map_location=device
            ).items()

            if not k.startswith('classifier')
        }

        model.speech_enc.load_state_dict(
            st,
            strict=False
        )

        print(
            f'Loaded speech encoder from {speech_ckpt}'
        )

    # ━━━━━━━━━━━━━━━━━━━━━
    # LOAD DISTILBERT TEXT MODEL
    # ━━━━━━━━━━━━━━━━━━━━━

    if os.path.exists(text_ckpt):

        st = {

            k: v

            for k, v in torch.load(

                text_ckpt,

                map_location=device
            ).items()

            if not k.startswith('classifier')
        }

        model.text_enc.load_state_dict(
            st,
            strict=False
        )

        print(
            f'Loaded DistilBERT text encoder from {text_ckpt}'
        )

    # ━━━━━━━━━━━━━━━━━━━━━
    # LOSS
    # ━━━━━━━━━━━━━━━━━━━━━

    crit = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # ━━━━━━━━━━━━━━━━━━━━━
    # OPTIMIZER
    # ━━━━━━━━━━━━━━━━━━━━━

    bert_p = list(
        model.text_enc.bert.parameters()
    )

    other_p = [

        p for p in model.parameters()

        if id(p) not in {

            id(bp) for bp in bert_p
        }
    ]

    opt = torch.optim.AdamW(

        [

            {
                'params': bert_p,
                'lr': lr * 0.1
            },

            {
                'params': other_p,
                'lr': lr
            }
        ],

        weight_decay=weight_decay
    )

    # ━━━━━━━━━━━━━━━━━━━━━
    # SCHEDULER
    # ━━━━━━━━━━━━━━━━━━━━━

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(

        opt,

        T_max=epochs
    )

    hist = {

        'train_loss': [],

        'val_loss': [],

        'train_acc': [],

        'val_acc': []
    }

    best = 0.0

    no_improve = 0

    spath = os.path.join(

        results_dir,

        'fusion_best_model.pt'
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    print(
        '\n=== FUSION TRAINING '
        '(Speech + DistilBERT Text) ==='
    )

    # ━━━━━━━━━━━━━━━━━━━━━
    # TRAIN LOOP
    # ━━━━━━━━━━━━━━━━━━━━━

    for ep in range(1, epochs + 1):

        tl, ta = train_epoch(

            model,

            trl,

            opt,

            crit,

            device
        )

        vl_, va_, _, _ = evaluate(

            model,

            vl,

            crit,

            device
        )

        sched.step()

        hist['train_loss'].append(tl)

        hist['val_loss'].append(vl_)

        hist['train_acc'].append(ta)

        hist['val_acc'].append(va_)

        print(

            f'Ep {ep:03d}/{epochs}'

            f' | TrLoss {tl:.4f}'

            f' TrAcc {ta:.4f}'

            f' | VaLoss {vl_:.4f}'

            f' VaAcc {va_:.4f}'
        )

        if va_ > best:

            best = va_

            no_improve = 0

            if save_best:

                torch.save(
                    model.state_dict(),
                    spath
                )

                print(
                    f'  Saved best ({best:.4f})'
                )

        else:

            no_improve += 1

            if (

                early_stopping_patience > 0

                and

                no_improve >= early_stopping_patience
            ):

                print(
                    f'\nEarly stopping at epoch {ep}'
                )

                break

    # ━━━━━━━━━━━━━━━━━━━━━
    # LOAD BEST MODEL
    # ━━━━━━━━━━━━━━━━━━━━━

    if save_best and os.path.exists(spath):

        model.load_state_dict(

            torch.load(

                spath,

                map_location=device
            )
        )

    # ━━━━━━━━━━━━━━━━━━━━━
    # TEST
    # ━━━━━━━━━━━━━━━━━━━━━

    _, acc, yp, yt = evaluate(

        model,

        tel,

        crit,

        device
    )

    print(
        f'\nTest Accuracy: '
        f'{acc:.4f} ({acc*100:.2f}%)'
    )

    save_results(

        results_dir,

        'fusion',

        yt,

        yp,

        hist
    )

    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':

    import argparse

    p = argparse.ArgumentParser()

    p.add_argument(
        '--data_root',
        type=str,
        required=True
    )

    p.add_argument(
        '--speech_ckpt',
        type=str,
        default='Results/speech_best_model.pt'
    )

    p.add_argument(
        '--text_ckpt',
        type=str,
        default='Results/text_best_model.pt'
    )

    p.add_argument(
        '--epochs',
        type=int,
        default=30
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=8
    )

    p.add_argument(
        '--lr',
        type=float,
        default=2e-5
    )

    p.add_argument(
        '--weight_decay',
        type=float,
        default=0.01
    )

    p.add_argument(
        '--results_dir',
        type=str,
        default='Results'
    )

    p.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5
    )

    a = p.parse_args()

    train(

        a.data_root,

        a.speech_ckpt,

        a.text_ckpt,

        a.epochs,

        a.batch_size,

        a.lr,

        a.weight_decay,

        a.results_dir,

        early_stopping_patience=
        a.early_stopping_patience
    )