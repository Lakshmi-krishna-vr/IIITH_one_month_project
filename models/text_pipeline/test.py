import os
import sys
import argparse
import torch

from torch.utils.data import DataLoader

from transformers import DistilBertTokenizer

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

from utils import (
    load_tess_dataset,
    get_splits,
    save_results,
    set_seed,
    get_device,
    IDX_TO_EMOTION,
    EMOTION_LABELS
)

from models.text_pipeline.train import (
    TextEmotionModel,
    TextDataset
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SINGLE WORD PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_word(
    word,
    model,
    tokenizer,
    device,
    max_len=32
):

    enc = tokenizer(
        word.lower(),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    ids = enc['input_ids'].to(device)

    attn = enc['attention_mask'].to(device)

    model.eval()

    with torch.no_grad():

        outputs = model(ids, attn)

        probs = torch.softmax(
            outputs,
            dim=1
        ).squeeze().cpu().numpy()

    pred = probs.argmax()

    print(f'\nInput text        : "{word}"')

    print(
        f'Predicted emotion : '
        f'{IDX_TO_EMOTION[pred]}'
    )

    print('\nEmotion Probabilities:\n')

    for i, p in enumerate(probs):

        bar = "█" * int(p * 30)

        print(
            f'{IDX_TO_EMOTION[i]:<20}'
            f'{p:.4f}   {bar}'
        )

    return IDX_TO_EMOTION[pred]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_root',
        type=str,
        default='data/TESS Toronto emotional speech set data'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='Results/text_best_model.pt'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='Results'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--sentence',
        type=str,
        default=None,
        help='Single text/sentence prediction'
    )

    args = parser.parse_args()

    # DEVICE

    set_seed(42)

    device = get_device()

    print(f'Using device: {device}')

    # TOKENIZER

    tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased'
    )

    # MODEL

    model = TextEmotionModel().to(device)

    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=device
        )
    )

    print(f'\nLoaded model from:\n{args.model_path}')

    # SINGLE WORD MODE

    if args.sentence:

        predict_word(
            args.sentence,
            model,
            tokenizer,
            device
        )

        return

    # LOAD DATASET

    print('\nLoading dataset ...')

    df = load_tess_dataset(args.data_root)

    _, _, test_df = get_splits(df)

    print(f'Test samples: {len(test_df)}')

    # DATALOADER

    loader = DataLoader(
        TextDataset(test_df, tokenizer),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # TESTING

    all_preds = []

    all_labels = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            input_ids = batch['input_ids'].to(device)

            attention_mask = batch[
                'attention_mask'
            ].to(device)

            labels = batch['label']

            outputs = model(
                input_ids,
                attention_mask
            )

            preds = outputs.argmax(1)

            all_preds.extend(
                preds.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    # ACCURACY

    acc = sum(
        p == l
        for p, l in zip(all_preds, all_labels)
    ) / len(all_labels)

    print(
        f'\nTest Accuracy: '
        f'{acc:.4f} ({acc*100:.2f}%)'
    )

    # SAVE RESULTS

    save_results(
        args.results_dir,
        'text_test',
        all_labels,
        all_preds
    )

    print(
        f'\nResults saved to: '
        f'{args.results_dir}'
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':

    main()